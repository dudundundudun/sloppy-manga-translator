use crate::preprocess::preprocess_nar_crop;
use image::RgbImage;
use miko_ort::ort::session::{Session, SessionInputValue};
use miko_ort::ort::value::{Tensor, TensorElementType, ValueType};
use miko_ort::{create_session, ExecutionProviderKind, OrtError};
use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NarError {
    #[error("Ошибка чтения словаря NAR-OCR: {0}")]
    VocabIo(#[from] std::io::Error),
    #[error("Ошибка JSON словаря NAR-OCR: {0}")]
    VocabJson(#[from] serde_json::Error),
    #[error("Ошибка ONNX Runtime в NAR-OCR: {0}")]
    Ort(#[from] OrtError),
    #[error("Ошибка сессии NAR-OCR: {0}")]
    Session(#[from] miko_ort::ort::Error),
}

#[derive(Debug, Deserialize)]
struct VocabJsonFile {
    characters: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct NarVocabulary {
    pub chars: Vec<String>,
    pub eos_id: usize,
}

impl NarVocabulary {
    pub fn load(path: &Path) -> Result<Self, NarError> {
        let raw = std::fs::read_to_string(path)?;
        let parsed: VocabJsonFile = serde_json::from_str(&raw)?;
        let eos_id = parsed.characters.len();
        Ok(Self {
            chars: parsed.characters,
            eos_id,
        })
    }

    pub fn from_chars(chars: Vec<String>) -> Self {
        let eos_id = chars.len();
        Self { chars, eos_id }
    }

    pub fn decode_ids(&self, ids: &[i64]) -> String {
        let mut out = String::new();
        for &raw_id in ids {
            if raw_id < 0 {
                continue;
            }
            let id = raw_id as usize;
            if id >= self.eos_id {
                break;
            }
            if let Some(ch) = self.chars.get(id) {
                out.push_str(ch);
            }
        }
        out
    }
}

pub struct NarRecognizer {
    session: Session,
    pub vocab: NarVocabulary,
    pub ep: ExecutionProviderKind,
}

impl NarRecognizer {
    pub fn load(onnx_path: &Path, vocab_path: &Path, prefer_gpu: bool) -> Result<Self, NarError> {
        let (session, ep) = create_session(onnx_path, prefer_gpu)?;
        let vocab = NarVocabulary::load(vocab_path)?;
        Ok(Self { session, vocab, ep })
    }

    fn make_tensor_for_inlet<'a>(
        name: &str,
        dtype: &ValueType,
        nar: &crate::preprocess::NarPreprocessOutput,
    ) -> Result<SessionInputValue<'a>, NarError> {
        let (ty, raw_shape) = match dtype {
            ValueType::Tensor { ty, shape, .. } => (*ty, &**shape),
            _ => {
                return Err(NarError::Ort(OrtError::UnsupportedInput {
                    name: name.to_string(),
                    reason: "Expected tensor inlet".to_string(),
                }))
            }
        };
        let is_batched = matches!(name, "P" | "V" | "R" | "C")
            && raw_shape.len() == if name == "P" { 3 } else { 2 }
            || matches!(name, "H" | "W") && raw_shape.first() == Some(&-1);

        match name {
            "P" => {
                let shape = if is_batched {
                    vec![1usize, 324, 588]
                } else {
                    vec![324usize, 588]
                };
                Ok(Tensor::from_array((shape, nar.p.clone()))?.into())
            }
            "V" => {
                let shape = if is_batched {
                    vec![1usize, 324]
                } else {
                    vec![324usize]
                };
                match ty {
                    TensorElementType::Bool => {
                        let data: Vec<bool> = nar.v.iter().map(|&x| x > 0.5).collect();
                        Ok(Tensor::from_array((shape, data))?.into())
                    }
                    _ => Ok(Tensor::from_array((shape, nar.v.clone()))?.into()),
                }
            }
            "R" | "C" => {
                let shape = if is_batched {
                    vec![1usize, 324]
                } else {
                    vec![324usize]
                };
                let src = if name == "R" { &nar.r } else { &nar.c };
                match ty {
                    TensorElementType::Int64 => {
                        let data: Vec<i64> = src.iter().map(|&x| x as i64).collect();
                        Ok(Tensor::from_array((shape, data))?.into())
                    }
                    _ => Ok(Tensor::from_array((shape, src.clone()))?.into()),
                }
            }
            "H" | "W" => {
                let shape = if raw_shape.is_empty() {
                    Vec::new()
                } else {
                    vec![1usize]
                };
                let val = if name == "H" { nar.h } else { nar.w };
                match ty {
                    TensorElementType::Int64 => {
                        Ok(Tensor::from_array((shape, vec![val as i64]))?.into())
                    }
                    _ => Ok(Tensor::from_array((shape, vec![val]))?.into()),
                }
            }
            _ => Err(NarError::Ort(OrtError::UnsupportedInput {
                name: name.to_string(),
                reason: "Unknown NAR input name".to_string(),
            })),
        }
    }

    pub fn recognize_crop(&mut self, crop: &RgbImage) -> Result<String, NarError> {
        let prep = preprocess_nar_crop(crop);
        let mut named_inputs: Vec<(String, SessionInputValue<'static>)> =
            Vec::with_capacity(self.session.inputs().len());
        for inlet in self.session.inputs() {
            let name = inlet.name().to_string();
            let val = Self::make_tensor_for_inlet(&name, inlet.dtype(), &prep)?;
            named_inputs.push((name, val));
        }
        let outputs = self.session.run(named_inputs)?;
        let out_val = outputs.values().next().ok_or_else(|| {
            NarError::Ort(OrtError::UnsupportedInput {
                name: "ids".to_string(),
                reason: "Missing output".to_string(),
            })
        })?;
        if let Ok((_, ids_i64)) = out_val.try_extract_tensor::<i64>() {
            return Ok(self.vocab.decode_ids(ids_i64));
        }
        let (_, ids_i32) = out_val.try_extract_tensor::<i32>()?;
        let ids: Vec<i64> = ids_i32.iter().map(|&x| x as i64).collect();
        Ok(self.vocab.decode_ids(&ids))
    }
}
