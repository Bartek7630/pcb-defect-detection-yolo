import gradio as gr
import torch
import spaces

old_load = torch.load
def unbricked_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return old_load(*args, **kwargs)
torch.load = unbricked_load

from ultralytics import YOLO

model = YOLO("best.pt")

@spaces.GPU
def predict_pcb(image):
    results = model.predict(image, conf=0.25, verbose=False)
    res_plotted = results[0].plot()
    return res_plotted[..., ::-1]

demo = gr.Interface(
    fn=predict_pcb,
    inputs=gr.Image(type="pil", label="Wgraj zdjęcie płytki PCB"),
    outputs=gr.Image(type="numpy", label="Wykryte defekty"),
    title="Automatyczna Inspekcja PCB",
    description="Wgraj zdjęcie płytki drukowanej, aby wykryć anomalie produkcyjne."
)

if __name__ == "__main__":
    demo.launch()