//! A Falcon checkpoint is loaded at the MLP width its config states.
//!
//! The MLP was built four times `hidden_size` wide whatever the config said.
//! The original Falcon configs omit `ffn_hidden_size` because that is their
//! width; a checkpoint that sets another one (`Intel/tiny-random-falcon`: 128
//! hidden, 256 inner) failed to load with a shape mismatch on `dense_h_to_4h`.

use candle::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::falcon::{Config, Falcon};

fn config(ffn_hidden_size: Option<usize>) -> Config {
    let mut text = String::from(
        r#"{
            "vocab_size": 64, "hidden_size": 128, "num_hidden_layers": 1,
            "num_attention_heads": 2, "layer_norm_epsilon": 1e-5, "initializer_range": 0.02,
            "use_cache": true, "bos_token_id": 1, "eos_token_id": 1,
            "hidden_dropout": 0.0, "attention_dropout": 0.0, "n_head_kv": null,
            "alibi": false, "new_decoder_architecture": false, "multi_query": true,
            "parallel_attn": true, "bias": false"#,
    );
    if let Some(width) = ffn_hidden_size {
        text.push_str(&format!(r#", "ffn_hidden_size": {width}"#));
    }
    text.push('}');
    serde_json::from_str(&text).unwrap()
}

/// The shape the model asked for its first MLP weight.
fn mlp_weight_shape(cfg: Config) -> Vec<usize> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    Falcon::load(vb, cfg).unwrap();
    let data = varmap.data().lock().unwrap();
    data["transformer.h.0.mlp.dense_h_to_4h.weight"].dims().to_vec()
}

#[test]
fn falcon_mlp_follows_ffn_hidden_size() {
    assert_eq!(mlp_weight_shape(config(Some(256))), vec![256, 128]);
    // Absent, it is four times the hidden size, as in the original Falcons.
    assert_eq!(mlp_weight_shape(config(None)), vec![512, 128]);
}
