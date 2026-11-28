use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativePickOutcome {
    Selected(PathBuf),
    Cancelled,
    Timeout,
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialogMode {
    Folder,
    ImageFile,
    SaveFolder,
}

impl DialogMode {
    pub fn as_cli_arg(self) -> &'static str {
        match self {
            DialogMode::Folder => "folder",
            DialogMode::ImageFile => "file",
            DialogMode::SaveFolder => "save_folder",
        }
    }

    pub fn from_cli_arg(s: &str) -> Option<Self> {
        match s.trim() {
            "folder" => Some(DialogMode::Folder),
            "file" => Some(DialogMode::ImageFile),
            "save_folder" => Some(DialogMode::SaveFolder),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivePickerCheck {
    NotRunning,
    StaleCleared,
    Focused,
    ResetKilled,
}

/// Перехватывает внутренний режим дочернего процесса `--pick-dialog <mode> <fg_hwnd> [initial_dir]`
/// на самом старте `main()` (до инициализации Tokio, SQLite и ONNX), чтобы окно Проводника
/// открывалось за ~10 мс.
pub fn try_run_cli_dialog_from_args() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let idx = args.iter().position(|a| a == "--pick-dialog")?;
    let mode_str = args.get(idx + 1).map(String::as_str).unwrap_or("folder");
    let mode = DialogMode::from_cli_arg(mode_str).unwrap_or(DialogMode::Folder);
    let fg_hwnd = args
        .get(idx + 2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let initial_dir = args
        .get(idx + 3)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);

    let outcome = run_inproc_dialog(mode, fg_hwnd, initial_dir.as_deref());
    let payload = match outcome {
        NativePickOutcome::Selected(p) => json!({
            "status": "selected",
            "path": p.to_string_lossy(),
        }),
        NativePickOutcome::Cancelled => json!({
            "status": "cancelled",
        }),
        NativePickOutcome::Timeout => json!({
            "status": "timeout",
        }),
        NativePickOutcome::Error(msg) => json!({
            "status": "error",
            "message": msg,
        }),
    };
    println!("{payload}");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    Some(0)
}

struct ChildPidGuard<'a> {
    slot: &'a Mutex<Option<u32>>,
}

impl<'a> Drop for ChildPidGuard<'a> {
    fn drop(&mut self) {
        if let Ok(mut g) = self.slot.lock() {
            *g = None;
        }
    }
}

pub fn spawn_picker_subprocess(
    mode: DialogMode,
    initial_dir: Option<&Path>,
    pid_slot: &Mutex<Option<u32>>,
    timeout: Duration,
) -> NativePickOutcome {
    #[cfg(not(windows))]
    {
        let _ = (mode, initial_dir, pid_slot, timeout);
        NativePickOutcome::Error("Нативный диалог выбора поддерживается только в Windows".to_string())
    }
    #[cfg(windows)]
    {
        use std::io::Read;
        use std::os::windows::process::CommandExt;
        use std::process::{Command, Stdio};

        const CREATE_NO_WINDOW: u32 = 0x0800_0000;

        let fg_hwnd = win32::prepare_parent_foreground_unlock();
        let exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => {
                return NativePickOutcome::Error(format!(
                    "Не удалось определить путь исполняемого файла: {e}"
                ));
            }
        };

        let init_arg = initial_dir
            .filter(|p| p.exists())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut cmd = Command::new(&exe);
        cmd.args([
            "--pick-dialog",
            mode.as_cli_arg(),
            &fg_hwnd.to_string(),
            &init_arg,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW);

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return NativePickOutcome::Error(format!(
                    "Не удалось запустить процесс системного диалога: {e}"
                ));
            }
        };

        let child_pid = child.id();
        win32::allow_foreground_for_pid(child_pid);
        if let Ok(mut g) = pid_slot.lock() {
            *g = Some(child_pid);
        }
        let _pid_guard = ChildPidGuard { slot: pid_slot };

        let stdout_handle = child.stdout.take().map(|mut out| {
            std::thread::spawn(move || {
                let mut buf = String::new();
                let _ = out.read_to_string(&mut buf);
                buf
            })
        });
        let stderr_handle = child.stderr.take().map(|mut err| {
            std::thread::spawn(move || {
                let mut buf = String::new();
                let _ = err.read_to_string(&mut buf);
                buf
            })
        });

        let started = Instant::now();
        let exit_status = loop {
            match child.try_wait() {
                Ok(Some(st)) => break Ok(st),
                Ok(None) => {
                    if started.elapsed() >= timeout {
                        win32::kill_process_tree(child_pid);
                        let _ = child.kill();
                        let _ = child.wait();
                        return NativePickOutcome::Timeout;
                    }
                    std::thread::sleep(Duration::from_millis(35));
                }
                Err(e) => break Err(e),
            }
        };

        let stdout_str = stdout_handle
            .and_then(|h| h.join().ok())
            .unwrap_or_default();
        let stderr_str = stderr_handle
            .and_then(|h| h.join().ok())
            .unwrap_or_default();

        if let Err(e) = exit_status {
            return NativePickOutcome::Error(format!("Ошибка ожидания диалога выбора: {e}"));
        }

        let trimmed = stdout_str.trim();
        if trimmed.is_empty() {
            if !stderr_str.trim().is_empty() {
                return NativePickOutcome::Error(stderr_str.trim().to_string());
            }
            return NativePickOutcome::Cancelled;
        }

        let last_line = trimmed.lines().last().unwrap_or(trimmed).trim();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(last_line) {
            match v.get("status").and_then(|s| s.as_str()).unwrap_or("") {
                "selected" => {
                    let raw = v.get("path").and_then(|s| s.as_str()).unwrap_or("").trim();
                    if raw.is_empty() {
                        NativePickOutcome::Cancelled
                    } else {
                        NativePickOutcome::Selected(PathBuf::from(raw))
                    }
                }
                "cancelled" => NativePickOutcome::Cancelled,
                "timeout" => NativePickOutcome::Timeout,
                "error" => {
                    let msg = v
                        .get("message")
                        .and_then(|s| s.as_str())
                        .unwrap_or("Системный диалог вернул ошибку")
                        .to_string();
                    NativePickOutcome::Error(msg)
                }
                _ => NativePickOutcome::Cancelled,
            }
        } else {
            NativePickOutcome::Selected(PathBuf::from(last_line))
        }
    }
}

pub fn focus_or_check_active_picker(
    pid_slot: &Mutex<Option<u32>>,
    force_reset: bool,
) -> ActivePickerCheck {
    let maybe_pid = pid_slot.lock().ok().and_then(|g| *g);
    let Some(pid) = maybe_pid else {
        return ActivePickerCheck::NotRunning;
    };

    #[cfg(windows)]
    {
        if !win32::is_process_alive(pid) {
            if let Ok(mut g) = pid_slot.lock() {
                if *g == Some(pid) {
                    *g = None;
                }
            }
            return ActivePickerCheck::StaleCleared;
        }
        if force_reset {
            win32::kill_process_tree(pid);
            if let Ok(mut g) = pid_slot.lock() {
                if *g == Some(pid) {
                    *g = None;
                }
            }
            return ActivePickerCheck::ResetKilled;
        }
        win32::focus_windows_of_pid(pid);
        ActivePickerCheck::Focused
    }
    #[cfg(not(windows))]
    {
        let _ = (pid, force_reset);
        ActivePickerCheck::NotRunning
    }
}

fn run_inproc_dialog(
    mode: DialogMode,
    fg_hwnd: usize,
    initial_dir: Option<&Path>,
) -> NativePickOutcome {
    #[cfg(windows)]
    {
        win32::show_explorer_file_dialog(mode, fg_hwnd, initial_dir)
    }
    #[cfg(not(windows))]
    {
        let _ = (mode, fg_hwnd, initial_dir);
        NativePickOutcome::Error("Нативный диалог выбора поддерживается только в Windows".to_string())
    }
}

#[cfg(windows)]
mod win32 {
    use super::{DialogMode, NativePickOutcome};
    use std::ffi::c_void;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::path::{Path, PathBuf};
    use std::ptr::{null, null_mut};

    type Hwnd = isize;
    type Hinstance = isize;
    type Hmenu = isize;
    type Hresult = i32;
    type Bool = i32;
    type Dword = u32;
    type Uint = u32;

    const ASFW_ANY: Dword = 0xFFFF_FFFF;
    const COINIT_APARTMENTTHREADED: Dword = 0x2;
    const CLSCTX_INPROC_SERVER: Dword = 0x1;

    const WS_POPUP: Dword = 0x8000_0000;
    const WS_EX_TOPMOST: Dword = 0x0000_0008;
    const WS_EX_APPWINDOW: Dword = 0x0004_0000;

    const SW_HIDE: i32 = 0;
    const SW_SHOW: i32 = 5;
    const SW_RESTORE: i32 = 9;
    const HWND_TOPMOST: Hwnd = -1;
    const SWP_NOSIZE: Uint = 0x0001;
    const SWP_NOMOVE: Uint = 0x0002;
    const SWP_SHOWWINDOW: Uint = 0x0040;
    const SM_CXSCREEN: i32 = 0;
    const SM_CYSCREEN: i32 = 1;
    const PM_REMOVE: Uint = 0x0001;

    const FOS_PICKFOLDERS: Dword = 0x0000_0020;
    const FOS_FORCEFILESYSTEM: Dword = 0x0000_0040;
    const FOS_PATHMUSTEXIST: Dword = 0x0000_0800;
    const FOS_FILEMUSTEXIST: Dword = 0x0000_1000;
    const FOS_NOCHANGEDIR: Dword = 0x0000_0008;
    const SIGDN_FILESYSPATH: Dword = 0x8005_8000;

    const HR_ERROR_CANCELLED: Hresult = 0x8007_04C7u32 as i32;
    const RPC_E_CHANGED_MODE: Hresult = 0x8001_0106u32 as i32;

    const PROCESS_QUERY_LIMITED_INFORMATION: Dword = 0x1000;
    const STILL_ACTIVE: Dword = 259;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub(super) struct Guid {
        pub data1: u32,
        pub data2: u16,
        pub data3: u16,
        pub data4: [u8; 8],
    }

    pub(super) const CLSID_FILE_OPEN_DIALOG: Guid = Guid {
        data1: 0xDC1C_5A9C,
        data2: 0xE88A,
        data3: 0x4DDE,
        data4: [0xA5, 0xA1, 0x60, 0xF8, 0x2A, 0x20, 0xAE, 0xF7],
    };

    pub(super) const IID_IFILE_DIALOG: Guid = Guid {
        data1: 0x42F8_5136,
        data2: 0xDB7E,
        data3: 0x439C,
        data4: [0x85, 0xF1, 0xE4, 0x07, 0x5D, 0x13, 0x5F, 0xC8],
    };

    pub(super) const IID_ISHELL_ITEM: Guid = Guid {
        data1: 0x4382_6D1E,
        data2: 0xE718,
        data3: 0x42EE,
        data4: [0xBC, 0x55, 0xA1, 0xE2, 0x61, 0xC3, 0x7B, 0xFE],
    };

    #[repr(C)]
    pub(super) struct ComdlgFilterspec {
        pub psz_name: *const u16,
        pub psz_spec: *const u16,
    }

    #[repr(C)]
    struct Point {
        x: i32,
        y: i32,
    }

    #[repr(C)]
    struct Msg {
        hwnd: Hwnd,
        message: Uint,
        w_param: usize,
        l_param: isize,
        time: Dword,
        pt: Point,
        l_private: Dword,
    }

    #[repr(C)]
    pub(super) struct IShellItem {
        pub lp_vtbl: *const IShellItemVtbl,
    }

    #[repr(C)]
    pub(super) struct IShellItemVtbl {
        pub query_interface:
            unsafe extern "system" fn(*mut IShellItem, *const Guid, *mut *mut c_void) -> Hresult,
        pub add_ref: unsafe extern "system" fn(*mut IShellItem) -> u32,
        pub release: unsafe extern "system" fn(*mut IShellItem) -> u32,
        pub bind_to_handler: unsafe extern "system" fn(
            *mut IShellItem,
            *mut c_void,
            *const Guid,
            *const Guid,
            *mut *mut c_void,
        ) -> Hresult,
        pub get_parent:
            unsafe extern "system" fn(*mut IShellItem, *mut *mut IShellItem) -> Hresult,
        pub get_display_name:
            unsafe extern "system" fn(*mut IShellItem, Dword, *mut *mut u16) -> Hresult,
        pub get_attributes:
            unsafe extern "system" fn(*mut IShellItem, Dword, *mut Dword) -> Hresult,
        pub compare:
            unsafe extern "system" fn(*mut IShellItem, *mut IShellItem, Dword, *mut i32) -> Hresult,
    }

    #[repr(C)]
    pub(super) struct IFileDialog {
        pub lp_vtbl: *const IFileDialogVtbl,
    }

    #[repr(C)]
    pub(super) struct IFileDialogVtbl {
        // IUnknown (0..=2)
        pub query_interface:
            unsafe extern "system" fn(*mut IFileDialog, *const Guid, *mut *mut c_void) -> Hresult,
        pub add_ref: unsafe extern "system" fn(*mut IFileDialog) -> u32,
        pub release: unsafe extern "system" fn(*mut IFileDialog) -> u32,
        // IModalWindow (3)
        pub show: unsafe extern "system" fn(*mut IFileDialog, Hwnd) -> Hresult,
        // IFileDialog (4..=26)
        pub set_file_types: unsafe extern "system" fn(
            *mut IFileDialog,
            Uint,
            *const ComdlgFilterspec,
        ) -> Hresult,
        pub set_file_type_index: unsafe extern "system" fn(*mut IFileDialog, Uint) -> Hresult,
        pub get_file_type_index:
            unsafe extern "system" fn(*mut IFileDialog, *mut Uint) -> Hresult,
        pub advise:
            unsafe extern "system" fn(*mut IFileDialog, *mut c_void, *mut Dword) -> Hresult,
        pub unadvise: unsafe extern "system" fn(*mut IFileDialog, Dword) -> Hresult,
        pub set_options: unsafe extern "system" fn(*mut IFileDialog, Dword) -> Hresult,
        pub get_options: unsafe extern "system" fn(*mut IFileDialog, *mut Dword) -> Hresult,
        pub set_default_folder:
            unsafe extern "system" fn(*mut IFileDialog, *mut IShellItem) -> Hresult,
        pub set_folder: unsafe extern "system" fn(*mut IFileDialog, *mut IShellItem) -> Hresult,
        pub get_folder:
            unsafe extern "system" fn(*mut IFileDialog, *mut *mut IShellItem) -> Hresult,
        pub get_current_selection:
            unsafe extern "system" fn(*mut IFileDialog, *mut *mut IShellItem) -> Hresult,
        pub set_file_name: unsafe extern "system" fn(*mut IFileDialog, *const u16) -> Hresult,
        pub get_file_name: unsafe extern "system" fn(*mut IFileDialog, *mut *mut u16) -> Hresult,
        pub set_title: unsafe extern "system" fn(*mut IFileDialog, *const u16) -> Hresult,
        pub set_ok_button_label:
            unsafe extern "system" fn(*mut IFileDialog, *const u16) -> Hresult,
        pub set_file_name_label:
            unsafe extern "system" fn(*mut IFileDialog, *const u16) -> Hresult,
        pub get_result:
            unsafe extern "system" fn(*mut IFileDialog, *mut *mut IShellItem) -> Hresult,
        pub add_place:
            unsafe extern "system" fn(*mut IFileDialog, *mut IShellItem, Dword) -> Hresult,
        pub set_default_extension:
            unsafe extern "system" fn(*mut IFileDialog, *const u16) -> Hresult,
        pub close: unsafe extern "system" fn(*mut IFileDialog, Hresult) -> Hresult,
        pub set_client_guid: unsafe extern "system" fn(*mut IFileDialog, *const Guid) -> Hresult,
        pub clear_client_data: unsafe extern "system" fn(*mut IFileDialog) -> Hresult,
        pub set_filter: unsafe extern "system" fn(*mut IFileDialog, *mut c_void) -> Hresult,
    }

    type EnumWindowsProc = unsafe extern "system" fn(Hwnd, isize) -> Bool;

    #[link(name = "user32")]
    extern "system" {
        fn SetProcessDPIAware() -> Bool;
        fn AllowSetForegroundWindow(dw_process_id: Dword) -> Bool;
        fn GetForegroundWindow() -> Hwnd;
        fn GetWindowThreadProcessId(hwnd: Hwnd, lpdw_process_id: *mut Dword) -> Dword;
        fn AttachThreadInput(id_attach: Dword, id_attach_to: Dword, f_attach: Bool) -> Bool;
        fn GetSystemMetrics(n_index: i32) -> i32;
        fn CreateWindowExW(
            dw_ex_style: Dword,
            lp_class_name: *const u16,
            lp_window_name: *const u16,
            dw_style: Dword,
            x: i32,
            y: i32,
            n_width: i32,
            n_height: i32,
            hwnd_parent: Hwnd,
            h_menu: Hmenu,
            h_instance: Hinstance,
            lp_param: *mut c_void,
        ) -> Hwnd;
        fn DestroyWindow(hwnd: Hwnd) -> Bool;
        fn ShowWindow(hwnd: Hwnd, n_cmd_show: i32) -> Bool;
        fn IsWindowVisible(hwnd: Hwnd) -> Bool;
        fn IsIconic(hwnd: Hwnd) -> Bool;
        fn SetWindowPos(
            hwnd: Hwnd,
            hwnd_insert_after: Hwnd,
            x: i32,
            y: i32,
            cx: i32,
            cy: i32,
            u_flags: Uint,
        ) -> Bool;
        fn BringWindowToTop(hwnd: Hwnd) -> Bool;
        fn SetForegroundWindow(hwnd: Hwnd) -> Bool;
        fn SetActiveWindow(hwnd: Hwnd) -> Hwnd;
        fn PeekMessageW(
            lp_msg: *mut Msg,
            hwnd: Hwnd,
            w_msg_filter_min: Uint,
            w_msg_filter_max: Uint,
            w_remove_msg: Uint,
        ) -> Bool;
        fn TranslateMessage(lp_msg: *const Msg) -> Bool;
        fn DispatchMessageW(lp_msg: *const Msg) -> isize;
        fn EnumWindows(lp_enum_func: EnumWindowsProc, l_param: isize) -> Bool;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThreadId() -> Dword;
        fn GetModuleHandleW(lp_module_name: *const u16) -> Hinstance;
        fn OpenProcess(dw_desired_access: Dword, b_inherit_handle: Bool, dw_process_id: Dword) -> isize;
        fn GetExitCodeProcess(h_process: isize, lp_exit_code: *mut Dword) -> Bool;
        fn CloseHandle(h_object: isize) -> Bool;
    }

    #[link(name = "ole32")]
    extern "system" {
        pub(super) fn CoInitializeEx(pv_reserved: *mut c_void, dw_co_init: Dword) -> Hresult;
        pub(super) fn CoUninitialize();
        pub(super) fn CoCreateInstance(
            rclsid: *const Guid,
            p_unk_outer: *mut c_void,
            dw_cls_context: Dword,
            riid: *const Guid,
            ppv: *mut *mut c_void,
        ) -> Hresult;
        pub(super) fn CoTaskMemFree(pv: *mut c_void);
    }

    #[link(name = "shell32")]
    extern "system" {
        pub(super) fn SHCreateItemFromParsingName(
            psz_path: *const u16,
            pbc: *mut c_void,
            riid: *const Guid,
            ppv: *mut *mut c_void,
        ) -> Hresult;
    }

    pub(super) fn to_wide_null(s: &str) -> Vec<u16> {
        std::ffi::OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    pub(super) unsafe fn wide_ptr_to_string(ptr: *const u16) -> String {
        if ptr.is_null() {
            return String::new();
        }
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = std::slice::from_raw_parts(ptr, len);
        std::ffi::OsString::from_wide(slice)
            .to_string_lossy()
            .into_owned()
    }

    pub fn prepare_parent_foreground_unlock() -> usize {
        unsafe {
            let _ = AllowSetForegroundWindow(ASFW_ANY);
            GetForegroundWindow() as usize
        }
    }

    pub fn allow_foreground_for_pid(pid: u32) {
        unsafe {
            let _ = AllowSetForegroundWindow(pid);
        }
    }

    pub fn is_process_alive(pid: u32) -> bool {
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle == 0 {
                return false;
            }
            let mut exit_code: Dword = 0;
            let ok = GetExitCodeProcess(handle, &mut exit_code);
            let _ = CloseHandle(handle);
            ok != 0 && exit_code == STILL_ACTIVE
        }
    }

    pub fn kill_process_tree(pid: u32) {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(CREATE_NO_WINDOW)
            .output();
    }

    unsafe extern "system" fn focus_pid_enum_cb(hwnd: Hwnd, l_param: isize) -> Bool {
        let target_pid = l_param as Dword;
        let mut wnd_pid: Dword = 0;
        let wnd_tid = GetWindowThreadProcessId(hwnd, &mut wnd_pid);
        if wnd_pid == target_pid && IsWindowVisible(hwnd) != 0 {
            let fg_hwnd = GetForegroundWindow();
            let fg_tid = if fg_hwnd != 0 {
                GetWindowThreadProcessId(fg_hwnd, null_mut())
            } else {
                0
            };
            let cur_tid = GetCurrentThreadId();
            let attached_fg = if fg_tid != 0 && fg_tid != cur_tid {
                AttachThreadInput(fg_tid, cur_tid, 1) != 0
            } else {
                false
            };
            let attached_wnd = if wnd_tid != 0 && wnd_tid != cur_tid {
                AttachThreadInput(wnd_tid, cur_tid, 1) != 0
            } else {
                false
            };

            if IsIconic(hwnd) != 0 {
                let _ = ShowWindow(hwnd, SW_RESTORE);
            } else {
                let _ = ShowWindow(hwnd, SW_SHOW);
            }
            let _ = SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
            );
            let _ = BringWindowToTop(hwnd);
            let _ = SetForegroundWindow(hwnd);

            if attached_wnd {
                let _ = AttachThreadInput(wnd_tid, cur_tid, 0);
            }
            if attached_fg {
                let _ = AttachThreadInput(fg_tid, cur_tid, 0);
            }
        }
        1
    }

    pub fn focus_windows_of_pid(pid: u32) {
        unsafe {
            let _ = AllowSetForegroundWindow(ASFW_ANY);
            let _ = AllowSetForegroundWindow(pid);
            let _ = EnumWindows(focus_pid_enum_cb, pid as isize);
        }
    }

    unsafe fn pump_pending_messages() {
        let mut msg: Msg = std::mem::zeroed();
        for _ in 0..32 {
            if PeekMessageW(&mut msg, 0, 0, 0, PM_REMOVE) == 0 {
                break;
            }
            let _ = TranslateMessage(&msg);
            let _ = DispatchMessageW(&msg);
        }
    }

    unsafe fn create_topmost_owner_window(passed_fg_hwnd: usize) -> Hwnd {
        let sw = GetSystemMetrics(SM_CXSCREEN).max(800);
        let sh = GetSystemMetrics(SM_CYSCREEN).max(600);
        let cx = sw / 2;
        let cy = sh / 2;

        let cls = to_wide_null("STATIC");
        let title = to_wide_null("Miko-chan Manga Studio");
        let hinst = GetModuleHandleW(null());

        let hwnd = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_APPWINDOW,
            cls.as_ptr(),
            title.as_ptr(),
            WS_POPUP,
            cx,
            cy,
            1,
            1,
            0,
            0,
            hinst,
            null_mut(),
        );
        if hwnd == 0 {
            return 0;
        }

        let live_fg = GetForegroundWindow();
        let fg_hwnd = if live_fg != 0 {
            live_fg
        } else {
            passed_fg_hwnd as Hwnd
        };
        let fg_tid = if fg_hwnd != 0 {
            GetWindowThreadProcessId(fg_hwnd, null_mut())
        } else {
            0
        };
        let cur_tid = GetCurrentThreadId();
        let attached = if fg_tid != 0 && fg_tid != cur_tid {
            AttachThreadInput(fg_tid, cur_tid, 1) != 0
        } else {
            false
        };

        // Первый вызов ShowWindow сбрасывает возможный унаследованный флаг STARTF_USESHOWWINDOW (SW_HIDE)
        // от фонового родительского процесса; второй гарантированно делает окно видимым.
        let _ = ShowWindow(hwnd, SW_HIDE);
        let _ = ShowWindow(hwnd, SW_SHOW);
        let _ = SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            cx,
            cy,
            1,
            1,
            SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW,
        );
        let _ = BringWindowToTop(hwnd);
        let _ = SetForegroundWindow(hwnd);
        let _ = SetActiveWindow(hwnd);
        pump_pending_messages();

        if attached {
            let _ = AttachThreadInput(fg_tid, cur_tid, 0);
        }

        hwnd
    }

    pub fn show_explorer_file_dialog(
        mode: DialogMode,
        fg_hwnd: usize,
        initial_dir: Option<&Path>,
    ) -> NativePickOutcome {
        unsafe {
            let _ = SetProcessDPIAware();
            let hr_init = CoInitializeEx(null_mut(), COINIT_APARTMENTTHREADED);
            let should_uninit = hr_init >= 0;
            if hr_init < 0 && hr_init != RPC_E_CHANGED_MODE {
                return NativePickOutcome::Error(format!(
                    "CoInitializeEx failed: HRESULT 0x{:08X}",
                    hr_init as u32
                ));
            }

            let owner_hwnd = create_topmost_owner_window(fg_hwnd);

            let mut dlg_ptr: *mut c_void = null_mut();
            let hr_create = CoCreateInstance(
                &CLSID_FILE_OPEN_DIALOG,
                null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IFILE_DIALOG,
                &mut dlg_ptr,
            );
            if hr_create < 0 || dlg_ptr.is_null() {
                if owner_hwnd != 0 {
                    let _ = DestroyWindow(owner_hwnd);
                }
                if should_uninit {
                    CoUninitialize();
                }
                return NativePickOutcome::Error(format!(
                    "CoCreateInstance(FileOpenDialog) failed: HRESULT 0x{:08X}",
                    hr_create as u32
                ));
            }

            let dlg = dlg_ptr as *mut IFileDialog;
            let vtbl = &*(*dlg).lp_vtbl;

            let mut opts: Dword = 0;
            let _ = (vtbl.get_options)(dlg, &mut opts);
            opts |= FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR;

            match mode {
                DialogMode::Folder | DialogMode::SaveFolder => {
                    opts |= FOS_PICKFOLDERS | FOS_PATHMUSTEXIST;
                    let _ = (vtbl.set_options)(dlg, opts);
                    let title = match mode {
                        DialogMode::Folder => {
                            to_wide_null("Miko-chan Studio — Выберите папку с главой манги")
                        }
                        DialogMode::SaveFolder => {
                            to_wide_null("Miko-chan Studio — Выберите папку сохранения переводов")
                        }
                        DialogMode::ImageFile => unreachable!(),
                    };
                    let ok_lbl = to_wide_null("Выбрать папку");
                    let _ = (vtbl.set_title)(dlg, title.as_ptr());
                    let _ = (vtbl.set_ok_button_label)(dlg, ok_lbl.as_ptr());
                }
                DialogMode::ImageFile => {
                    opts |= FOS_PATHMUSTEXIST | FOS_FILEMUSTEXIST;
                    let _ = (vtbl.set_options)(dlg, opts);
                    let title = to_wide_null("Miko-chan Studio — Выберите страницу манги");
                    let _ = (vtbl.set_title)(dlg, title.as_ptr());

                    let f1_name = to_wide_null("Страницы манги (*.png;*.jpg;*.jpeg;*.webp;*.bmp)");
                    let f1_spec = to_wide_null("*.png;*.jpg;*.jpeg;*.webp;*.bmp");
                    let f2_name = to_wide_null("Все файлы (*.*)");
                    let f2_spec = to_wide_null("*.*");
                    let specs = [
                        ComdlgFilterspec {
                            psz_name: f1_name.as_ptr(),
                            psz_spec: f1_spec.as_ptr(),
                        },
                        ComdlgFilterspec {
                            psz_name: f2_name.as_ptr(),
                            psz_spec: f2_spec.as_ptr(),
                        },
                    ];
                    let _ = (vtbl.set_file_types)(dlg, specs.len() as Uint, specs.as_ptr());
                    let _ = (vtbl.set_file_type_index)(dlg, 1);
                }
            }

            if let Some(dir) = initial_dir.filter(|p| p.is_dir()) {
                let wide_dir = to_wide_null(&dir.to_string_lossy());
                let mut item_ptr: *mut c_void = null_mut();
                if SHCreateItemFromParsingName(
                    wide_dir.as_ptr(),
                    null_mut(),
                    &IID_ISHELL_ITEM,
                    &mut item_ptr,
                ) >= 0
                    && !item_ptr.is_null()
                {
                    let item = item_ptr as *mut IShellItem;
                    let _ = (vtbl.set_default_folder)(dlg, item);
                    let _ = ((*(*item).lp_vtbl).release)(item);
                }
            }

            let hr_show = (vtbl.show)(dlg, owner_hwnd);

            let outcome = if hr_show >= 0 {
                let mut res_item: *mut IShellItem = null_mut();
                let hr_res = (vtbl.get_result)(dlg, &mut res_item);
                if hr_res >= 0 && !res_item.is_null() {
                    let mut psz_path: *mut u16 = null_mut();
                    let hr_name =
                        ((*(*res_item).lp_vtbl).get_display_name)(res_item, SIGDN_FILESYSPATH, &mut psz_path);
                    let picked_str = if hr_name >= 0 && !psz_path.is_null() {
                        let s = wide_ptr_to_string(psz_path);
                        CoTaskMemFree(psz_path as *mut c_void);
                        Some(s)
                    } else {
                        None
                    };
                    let _ = ((*(*res_item).lp_vtbl).release)(res_item);
                    match picked_str {
                        Some(s) if !s.trim().is_empty() => {
                            NativePickOutcome::Selected(PathBuf::from(s.trim()))
                        }
                        _ => NativePickOutcome::Cancelled,
                    }
                } else {
                    NativePickOutcome::Cancelled
                }
            } else if hr_show == HR_ERROR_CANCELLED {
                NativePickOutcome::Cancelled
            } else {
                NativePickOutcome::Error(format!(
                    "Системный диалог вернул ошибку (HRESULT 0x{:08X})",
                    hr_show as u32
                ))
            };

            let _ = (vtbl.release)(dlg);
            if owner_hwnd != 0 {
                let _ = DestroyWindow(owner_hwnd);
            }
            if should_uninit {
                CoUninitialize();
            }
            outcome
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::win32::*;
    use std::ffi::c_void;
    use std::ptr::null_mut;

    #[test]
    fn test_com_file_open_dialog_vtable_and_shell_item() {
        unsafe {
            let hr_init = CoInitializeEx(null_mut(), 0x2);
            let should_uninit = hr_init >= 0;

            let mut dlg_ptr: *mut c_void = null_mut();
            let hr = CoCreateInstance(
                &CLSID_FILE_OPEN_DIALOG,
                null_mut(),
                0x1,
                &IID_IFILE_DIALOG,
                &mut dlg_ptr,
            );
            assert!(
                hr >= 0 && !dlg_ptr.is_null(),
                "CoCreateInstance(FileOpenDialog) must succeed: 0x{:08X}",
                hr as u32
            );

            let dlg = dlg_ptr as *mut IFileDialog;
            let vtbl = &*(*dlg).lp_vtbl;

            let mut opts: u32 = 0;
            assert!((vtbl.get_options)(dlg, &mut opts) >= 0);
            let want = opts | 0x20 | 0x40 | 0x800; // FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
            assert!((vtbl.set_options)(dlg, want) >= 0);
            let mut opts_after: u32 = 0;
            assert!((vtbl.get_options)(dlg, &mut opts_after) >= 0);
            assert_ne!(opts_after & 0x20, 0, "FOS_PICKFOLDERS must be set");

            let title = to_wide_null("Miko-chan VTable Check");
            assert!((vtbl.set_title)(dlg, title.as_ptr()) >= 0);
            let ok_lbl = to_wide_null("Выбрать папку");
            assert!((vtbl.set_ok_button_label)(dlg, ok_lbl.as_ptr()) >= 0);

            let temp_dir = std::env::temp_dir();
            let canon_temp = std::fs::canonicalize(&temp_dir).unwrap_or(temp_dir);
            let clean_temp = canon_temp
                .to_string_lossy()
                .trim_start_matches(r"\\?\")
                .to_string();
            let wide_temp = to_wide_null(&clean_temp);
            let mut item_ptr: *mut c_void = null_mut();
            let hr_item = SHCreateItemFromParsingName(
                wide_temp.as_ptr(),
                null_mut(),
                &IID_ISHELL_ITEM,
                &mut item_ptr,
            );
            assert!(
                hr_item >= 0 && !item_ptr.is_null(),
                "SHCreateItemFromParsingName failed: 0x{:08X}",
                hr_item as u32
            );
            let item = item_ptr as *mut IShellItem;
            assert!((vtbl.set_default_folder)(dlg, item) >= 0);

            let mut psz: *mut u16 = null_mut();
            let hr_disp = ((*(*item).lp_vtbl).get_display_name)(item, 0x8005_8000, &mut psz);
            assert!(hr_disp >= 0 && !psz.is_null());
            let roundtrip = wide_ptr_to_string(psz);
            CoTaskMemFree(psz as *mut c_void);
            assert_eq!(
                roundtrip.to_ascii_lowercase(),
                clean_temp.to_ascii_lowercase()
            );

            let _ = ((*(*item).lp_vtbl).release)(item);
            let _ = (vtbl.release)(dlg);

            if should_uninit {
                CoUninitialize();
            }
        }
    }
}
