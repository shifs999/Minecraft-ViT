# ViT-MAE

---

* This project is a basic implementation of Vision Transformer (ViT) model trained on Minecraft gameplay images using PyTorch. It uses unsupervised learning techniques, including masked autoencoding (MAE), to learn visual representations without labeled data.
* MAE doesn't rely on labeled data for training. Instead, it uses the image itself as both the input and the target, with a masking strategy to create a self-supervisory learning process. 
* The model encodes and reconstructs masked image patches, capturing the structure and style of Minecraft scenes.



## In order to generate the output run:

```bash
  python ModelTraining.py
```


## Input

![Alt text](https://github.com/user-attachments/assets/a474d057-3dec-4f80-ba5c-1f897f5e899c "Masked Input")

## Output

![Alt text](https://github.com/user-attachments/assets/8b8c973a-0d0f-4390-840e-3cd59142480c "Generated Output")

---
## Contact

For any queries or collaborations, feel free to reach me out at *saizen777999@gmail.com*
