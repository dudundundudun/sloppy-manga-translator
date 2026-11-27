use crate::detectors::BubbleDetection;

/// Порядок чтения бабблов на странице: классический манга RTL (справа налево, сверху вниз в ряду) или LTR.
pub fn sort_detections_reading_order(dets: &mut [BubbleDetection], rtl: bool) {
    if dets.len() <= 1 {
        return;
    }
    dets.sort_by(|a, b| {
        let ay_mid = (a.bbox[1] + a.bbox[3]) as f32 * 0.5;
        let by_mid = (b.bbox[1] + b.bbox[3]) as f32 * 0.5;
        let ah = (a.bbox[3] - a.bbox[1]).max(1) as f32;
        let bh = (b.bbox[3] - b.bbox[1]).max(1) as f32;

        // Если бабблы пересекаются по вертикали или близки в одной строке, сортируем горизонтально (RTL / LTR).
        let y_overlap = (a.bbox[3].min(b.bbox[3]) - a.bbox[1].max(b.bbox[1])) as f32;
        let min_h = ah.min(bh);
        let same_row = y_overlap > min_h * 0.35 || (ay_mid - by_mid).abs() < min_h * 0.40;
        if same_row {
            let ax_mid = (a.bbox[0] + a.bbox[2]) / 2;
            let bx_mid = (b.bbox[0] + b.bbox[2]) / 2;
            if rtl {
                bx_mid.cmp(&ax_mid)
            } else {
                ax_mid.cmp(&bx_mid)
            }
        } else {
            a.bbox[1].cmp(&b.bbox[1])
        }
    });
}
