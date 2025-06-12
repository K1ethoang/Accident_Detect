import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Use {device}')

if __name__ == '__main__':
    model = YOLO('./yolo11x.pt')
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=10,  
        device=device,
        batch=4,
        workers=2,
        save_period=2,
        project='runs/train',
        name='v11-x',
        # resume=True
    )
