import pytest
import torch

from src.efficient_transformer import CausalSelfAttention, TinyDecoderLM


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_attention_shape_and_finite(kv_heads):
    torch.manual_seed(0)
    layer = CausalSelfAttention(32, 4, kv_heads).eval()
    y = layer(torch.randn(2, 9, 32))
    assert y.shape == (2, 9, 32)
    assert torch.isfinite(y).all()


def test_invalid_head_ratio_rejected():
    with pytest.raises(ValueError):
        CausalSelfAttention(32, 4, 3)


def test_invalid_gqa_mode_rejected():
    with pytest.raises(ValueError):
        CausalSelfAttention(32, 4, 2, gqa_mode="fastest")


def test_native_and_expanded_gqa_equivalent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    native = CausalSelfAttention(32, 4, 2, gqa_mode="native").to(device).eval()
    expanded = CausalSelfAttention(32, 4, 2, gqa_mode="expand").to(device).eval()
    expanded.load_state_dict(native.state_dict())
    x = torch.randn(2, 9, 32, device=device)
    with torch.inference_mode():
        y_native = native(x)
        y_expanded = expanded(x)
    torch.testing.assert_close(y_native, y_expanded, rtol=2e-4, atol=2e-5)


def test_lm_and_generation():
    model = TinyDecoderLM(
        40, max_seq_len=16, dim=32, num_layers=1,
        num_q_heads=4, num_kv_heads=2
    ).eval()
    ids = torch.randint(0, 40, (2, 6))
    assert model(ids).shape == (2, 6, 40)
    assert model.generate(ids[:, :3], 2).shape == (2, 5)
