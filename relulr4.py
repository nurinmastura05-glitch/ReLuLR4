import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function", page_icon="🧠", layout="centered")

def relu(x):
    """Compute the ReLU activation function."""
    return np.maximum(0, x)

st.title("ReLU Activation Function")
st.write("ReLU (Rectified Linear Unit) returns zero for negative inputs and returns the input itself for positive inputs.")

st.latex(r"f(x)=\\max(0,x)")

st.subheader("Interactive visualisation")
x_min = st.slider("Minimum x-value", -20.0, 0.0, -10.0, 0.5)
x_max = st.slider("Maximum x-value", 0.0, 20.0, 10.0, 0.5)

if x_min >= x_max:
    st.error("The minimum x-value must be smaller than the maximum x-value.")
else:
    x = np.linspace(x_min, x_max, 500)
    y = relu(x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, linewidth=2.5, label="ReLU")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output f(x)")
    ax.set_title("ReLU Activation Function")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.metric("Output at x = 0", f"{float(relu(np.array([0.0]))[0]):.4f}")

st.subheader("Key characteristics")
st.markdown("""
- **Formula:** f(x) = max(0, x)
- **Output range:** [0, infinity)
- **Behaviour:** Negative inputs become 0; positive inputs remain unchanged.
- **Advantage:** Computationally simple and helps reduce the vanishing-gradient problem for positive inputs.
- **Limitation:** Neurons can become inactive when they repeatedly receive negative inputs (the dying ReLU problem).
""")

st.info("This application visualises how the activation function transforms input values. "
        "Adjust the sliders to explore its behaviour.")
