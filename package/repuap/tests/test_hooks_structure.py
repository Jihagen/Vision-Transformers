import torch
import torch.nn as nn

from repuap.hooks import (
    HookState, register_extract_hooks, register_inject_hooks,
    _inspect_model_structure, clear_structure_cache,
)


class _DummyDecoderLayer(nn.Module):
    """Mimics an HF decoder layer: forward returns a tuple whose first
    element is the hidden-state tensor, matching what the real hooks expect."""
    def forward(self, hidden_states):
        return (hidden_states,)


class _DummyLLM(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DummyDecoderLayer() for _ in range(n_layers)])


class _DummyVision(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DummyDecoderLayer() for _ in range(n_layers)])


class _DummyVLM(nn.Module):
    def __init__(self, n_llm: int = 4, n_vision: int = 3):
        super().__init__()
        self.language_model = _DummyLLM(n_llm)
        self.vision_model = _DummyVision(n_vision)


def test_inspect_model_structure_finds_both_stacks():
    clear_structure_cache()
    model = _DummyVLM(n_llm=4, n_vision=3)
    llm_layers, vision_layers, vision_path = _inspect_model_structure(model)
    assert len(llm_layers) == 4
    assert len(vision_layers) == 3
    assert vision_path == "vision_model.model.layers"


def test_inspect_model_structure_handles_llm_only():
    clear_structure_cache()
    model = nn.Module()
    model.language_model = _DummyLLM(2)
    llm_layers, vision_layers, vision_path = _inspect_model_structure(model)
    assert len(llm_layers) == 2
    assert vision_layers is None


def test_register_extract_hooks_captures_last_token_at_rating_step():
    clear_structure_cache()
    model = _DummyVLM(n_llm=2, n_vision=2)
    state = HookState()
    register_extract_hooks(model, state)

    hidden = torch.randn(1, 5, 8)  # (batch, seq, dim)
    state.phase = "rating_step"
    for layer in model.language_model.model.layers:
        layer(hidden)
    state.phase = "idle"

    assert "llm_layer_0_rating_token" in state.embeddings
    assert "llm_layer_1_rating_token" in state.embeddings
    captured = state.embeddings["llm_layer_0_rating_token"]
    assert captured.shape == (1, 8)  # last-token selector drops the seq dim
    assert torch.allclose(captured.float(), hidden[:, -1, :].float(), atol=1e-2)

    state.remove_hooks()


def test_register_extract_hooks_ignores_wrong_phase():
    clear_structure_cache()
    model = _DummyVLM(n_llm=2, n_vision=1)
    state = HookState()
    register_extract_hooks(model, state)

    hidden = torch.randn(1, 3, 8)
    state.phase = "vision_once"   # LLM hooks only fire on "rating_step"
    for layer in model.language_model.model.layers:
        layer(hidden)

    assert "llm_layer_0_rating_token" not in state.embeddings
    state.remove_hooks()


def test_register_inject_hooks_adds_alpha_times_vector():
    clear_structure_cache()
    model = _DummyVLM(n_llm=2, n_vision=1)
    state = HookState()
    vector = torch.ones(8).numpy()
    register_inject_hooks(model, state, {"llm_layer_0_rating_token": (vector, 2.0)})

    hidden = torch.zeros(1, 3, 8)
    out = model.language_model.model.layers[0](hidden)
    modified = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(modified, torch.full((1, 3, 8), 2.0))

    state.remove_hooks()
