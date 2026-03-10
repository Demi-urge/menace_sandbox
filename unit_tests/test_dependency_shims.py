from __future__ import annotations


def test_patch_logger_encoder_shim_exposes_encode_decode() -> None:
    import vector_service.patch_logger as pl

    encoded = pl._TokenizerShim().encode("abc")
    assert encoded == [97, 98, 99]
    assert pl._TokenizerShim().decode(encoded[:2]) == "ab"


def test_workflow_vectorizer_default_service_shim() -> None:
    import workflow_vectorizer as wv

    result = wv._DefaultVectorServiceShim().vectorise_and_store("workflow", "id", {"name": "n"})
    assert result == []


def test_vector_service_import_compat_shim_has_expected_methods() -> None:
    import vector_service as vs

    shim = vs._ImportCompatShim()
    assert shim.bootstrap("x", "y") is False
    assert hasattr(shim, "load_internal")


def test_bootstrap_placeholder_module_shim_has_expected_methods() -> None:
    import self_improvement.bootstrap_placeholder as bp

    shim = bp._BootstrapModuleShim()
    assert shim.advertise_broker_placeholder() == (None, None)
    assert shim.bootstrap_broker() == {}


def test_quick_fix_delegate_shim_callable() -> None:
    import menace_sandbox.quick_fix_engine as qfe

    assert qfe._FetchPatchDelegateShim()("patch-id") == {}


def test_embeddable_vec_metrics_shim_methods() -> None:
    import embeddable_db_mixin as edm

    shim = edm._VectorMetricsShim()
    assert shim.activate_persistence() is False
    assert shim.prepare_promotion() is False


def test_generative_provider_openai_pipeline_shims() -> None:
    import sandbox_runner.generative_stub_provider as gsp

    openai = gsp._OpenAIShim()
    openai.api_key = "k"
    assert openai.api_key == "k"
    assert gsp._PipelineShim()("prompt") == [{"generated_text": ""}]
