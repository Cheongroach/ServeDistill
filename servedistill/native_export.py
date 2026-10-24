"""Map merged Qwen3.5 language weights to the native vLLM wrapper format."""
import json
import shutil


def native_key(name):
    # Transformers may already serialize text weights using wrapper prefixes.
    if name.startswith("model.language_model."):
        return name
    if name.startswith("model."):
        return "model.language_model." + name.removeprefix("model.")
    if name.startswith("lm_head."):
        return name
    raise ValueError("Unexpected language checkpoint tensor: " + name)


def repackage_native(source, output):
    from safetensors.torch import load_file, save_file
    original = json.loads((source / "config.json").read_text())
    if original.get("model_type") != "qwen3_5":
        raise ValueError("Native vLLM packaging requires the original Qwen3.5 wrapper")
    tensor_count = 0
    for shard in sorted(output.glob("*.safetensors")):
        weights = load_file(shard)
        mapped = {native_key(key): tensor for key, tensor in weights.items()}
        temporary = shard.with_suffix(".repacked")
        save_file(mapped, temporary, metadata={"format": "pt"})
        temporary.replace(shard)
        tensor_count += len(mapped)
    index = output / "model.safetensors.index.json"
    if index.exists():
        value = json.loads(index.read_text())
        value["weight_map"] = {native_key(key): shard for key, shard in value["weight_map"].items()}
        index.write_text(json.dumps(value, indent=2) + "\n")
    original["dtype"] = "bfloat16"
    (output / "config.json").write_text(json.dumps(original, indent=2) + "\n")
    for name in ("preprocessor_config.json", "processor_config.json",
                 "video_preprocessor_config.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    return {"native_wrapper": original["architectures"],
            "language_tensors": tensor_count, "vision_weights_included": False,
            "required_server_flag": "--language-model-only"}
