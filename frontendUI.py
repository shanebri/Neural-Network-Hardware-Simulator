import pandas as pd
import streamlit as st

from analysis.api import estimate_model_runtime
from models.hardware_specs import get_gpu_spec, list_cpu_keys, list_gpu_keys
from models.mlp import SimpleMLP

st.set_page_config(page_title="Neural Network Hardware Simulator", layout="wide")
st.title("Neural Network Hardware Simulator")
st.caption(
    "Pick a model configuration and hardware profile to see estimated forward-pass runtime across batch sizes."
)

cpu_keys = list_cpu_keys()
gpu_keys = list_gpu_keys()

HARDWARE_TARGETS = {
    "CPU": {"key": "cpu", "profiles": cpu_keys, "default": "intel_i9_9900k"},
    "GPU": {"key": "gpu", "profiles": gpu_keys, "default": "nvidia_rtx_4070_ti"},
}


def build_mlp(input_dim: int, hidden1: int, hidden2: int, output_dim: int) -> SimpleMLP: #this is for future config options of models (planned)
    return SimpleMLP(
        input_dim=int(784),
        hidden1=int(128),
        hidden2=int(64),
        output_dim=int(10),
    )


@st.cache_data(show_spinner=False, hash_funcs={SimpleMLP: lambda model: repr(model)})
def run_batch_sweep(
    model: SimpleMLP,
    batches: list[int],
    hardware_key: str,
    hardware_profile: str,
    precision: str,
    include_bias: bool,
    include_activations: bool,
    activation_ops_per_element: int,
) -> pd.DataFrame:
    rows = []
    for batch in batches:
        result = estimate_model_runtime(
            model=model,
            batch_size=int(batch),
            hardware=hardware_key,
            hardware_profile=hardware_profile,
            precision=precision,
            include_bias=include_bias,
            include_activations=include_activations,
            activation_ops_per_element=int(activation_ops_per_element),
        )
        rows.append(
            {
                "hardware": result.hardware,
                "precision": result.precision,
                "batch_size": result.batch_size,
                "macs": result.macs,
                "flops": result.flops,
                "flop_rate": result.flop_rate,
                "est_runtime_s": result.est_runtime,
                "est_runtime_ms": result.est_runtime * 1_000,
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.subheader("Config")
    st.write("Select architecture and hardware configuration to see estimated runtime.")
    

    include_bias = st.checkbox("Include bias ops", value=True)
    include_activations = st.checkbox("Include activation ops", value=False)
    activation_ops_per_element = st.number_input(
        "Activation ops per element", min_value=0, value=1, step=1
    )

    st.subheader("Hardware")
    hardware_per_comparison = st.number_input(
        "How many hardware configurations to compare?", min_value=2, value=2, max_value=5, step=1
    )
    comparison_count = max(1, int(hardware_per_comparison))
    hardware_configs = []
    target_options = list(HARDWARE_TARGETS.keys())

    for idx in range(comparison_count):
        if idx == 1 and "GPU" in target_options:
            default_target = "GPU"
        else:
            default_target = target_options[0]
        target_label = st.selectbox(
            f"Target hardware {idx + 1}",
            target_options,
            index=target_options.index(default_target),
            key=f"target_label_{idx}",
        )
        target_info = HARDWARE_TARGETS[target_label]
        profiles = target_info["profiles"]
        default_profile = target_info["default"]
        default_profile_idx = profiles.index(default_profile) if default_profile in profiles else 0
        hardware_profile = st.selectbox(
            f"{target_label} profile {idx + 1}",
            profiles,
            index=default_profile_idx,
            key=f"profile_{idx}_{target_info['key']}",
        )
        precision = "fp32"
        if target_info["key"] == "gpu":
            gpu_spec = get_gpu_spec(hardware_profile)
            available_precisions = sorted(gpu_spec["theoretical_tflops"].keys())
            default_precision_idx = (
                available_precisions.index("fp32") if "fp32" in available_precisions else 0
            )
            precision = st.selectbox(
                f"Precision {idx + 1}",
                available_precisions,
                index=default_precision_idx,
                key=f"precision_{idx}",
            )
        hardware_configs.append(
            {
                "label": f"{target_label} {idx + 1}",
                "hardware_key": target_info["key"],
                "hardware_profile": hardware_profile,
                "precision": precision,
            }
        )

    config_keys = [
        (config["hardware_key"], config["hardware_profile"], config["precision"])
        for config in hardware_configs
    ]
    if len(config_keys) != len(set(config_keys)):
        st.warning(
            "Duplicate hardware selections detected. Consider choosing unique profiles for clearer comparisons."
        )

    primary_config = hardware_configs[0]
    hardware_key = primary_config["hardware_key"]
    hardware_profile = primary_config["hardware_profile"]
    precision = primary_config["precision"]


model = build_mlp( #when config is added set each variable to itself (user input defined earlier)
    input_dim=784,
    hidden1=128,
    hidden2=64,
    output_dim=10,
)

top_cols = st.columns([3, 1])
batch_size = top_cols[0].slider("Max Batch Size", min_value=6, max_value=2048, value=32, step=2)
model_name = top_cols[1].selectbox("Model", ["Simple MLP"]) #will be updated later when more models are added
sweep_start = 1
sweep_stop = batch_size
sweep_step = max(1, int(round(batch_size / 4)))
top_cols[0].caption(f"Sweep step: {sweep_step} (auto)")

try:
    single_estimate = estimate_model_runtime(
        model=model,
        batch_size=int(batch_size),
        hardware=hardware_key,
        hardware_profile=hardware_profile,
        precision=precision,
        include_bias=include_bias,
        include_activations=include_activations,
        activation_ops_per_element=int(activation_ops_per_element),
    )
except Exception as exc:
    st.error(f"Could not estimate runtime: {exc}")
    st.stop()

metric_cols = st.columns(3)
metric_cols[0].metric("Batch size", f"{single_estimate.batch_size}")
metric_cols[1].metric("MACs", f"{single_estimate.macs:,}")
metric_cols[2].metric("FLOPs", f"{single_estimate.flops:,}")


st.divider()
st.subheader("Batch sweep")
batch_values = list(range(int(sweep_start), int(sweep_stop) + 1, int(sweep_step)))

if not batch_values:
    st.info("Adjust the sweep controls to generate batch sizes.")
else:
    sweep_frames = []
    for config in hardware_configs:
        config_df = run_batch_sweep(
            model=model,
            batches=batch_values,
            hardware_key=config["hardware_key"],
            hardware_profile=config["hardware_profile"],
            precision=config["precision"],
            include_bias=include_bias,
            include_activations=include_activations,
            activation_ops_per_element=int(activation_ops_per_element),
        )
        config_label = f"{config['label']}: {config_df['hardware'].iloc[0]} ({config['precision']})"
        config_df["config"] = config_label
        sweep_frames.append(config_df)

    sweep_df = pd.concat(sweep_frames, ignore_index=True)
    chart_df = sweep_df.pivot(index="batch_size", columns="config", values="est_runtime_ms")
    st.line_chart(chart_df, height=360)
    st.dataframe(
        sweep_df.rename(columns={"est_runtime_s": "est_runtime_seconds"}),
        width="stretch",
    )
