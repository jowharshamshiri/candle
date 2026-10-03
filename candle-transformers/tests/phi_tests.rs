//! A phi model runs in the reduced-precision dtypes it is served in.
//!
//! `rope` needs the activations and its sin/cos tables in one dtype. The
//! tables were built in f32 whatever the weights were, so a phi loaded in f16
//! (as on CUDA) failed its first forward with "unsupported dtype for rope".

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::{Config, Model};

fn tiny_config() -> Config {
    serde_json::from_str(
        r#"{
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "hidden_act": "gelu_new",
            "max_position_embeddings": 64,
            "layer_norm_eps": 1e-5,
            "tie_word_embeddings": false,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "qk_layernorm": false
        }"#,
    )
    .unwrap()
}

fn forward_in(dtype: DType) -> candle::Result<Tensor> {
    let device = Device::Cpu;
    let vb = VarBuilder::zeros(dtype, &device);
    let mut model = Model::new(&tiny_config(), vb)?;
    let prompt = Tensor::new(&[[1u32, 2, 3, 4]], &device)?;
    let first = model.forward(&prompt)?;
    // And a step past the prompt, which reads the tables at an offset.
    let next = Tensor::new(&[[5u32]], &device)?;
    model.forward(&next)?;
    Ok(first)
}

// f16, as the model is served on CUDA. (bf16 is not checked here: the CPU
// backend has no bf16 matmul for any model.)
#[test]
fn phi_runs_in_f16() -> candle::Result<()> {
    for dtype in [DType::F32, DType::F16] {
        let logits = forward_in(dtype).map_err(|e| e.context(format!("phi in {dtype:?}")))?;
        assert_eq!(logits.dims(), &[1, 64], "phi in {dtype:?}");
    }
    Ok(())
}
