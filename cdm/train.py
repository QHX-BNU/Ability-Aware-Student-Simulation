import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import numpy as np
import json
from pathlib import Path
from data_loader import TrainDataLoader
from model import Net


# Dataset metadata (adjust as needed)
exer_n = 212 # 212 questions in total
knowledge_n = 93 # 93 knowledge components in total
student_n = 1299 # 1299 students in total

# Training hyperparameters (specified directly; no CLI)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epoch_n = 5   # epochs (set manually)

print(f"Using device: {device}")


# Project root: .../Ability-Aware-Student-Simulation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "cdm" / "model"
EMB_DIR = PROJECT_ROOT / "cdm" / "stu_embedding"

def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    print('training model...')

    loss_function = nn.NLLLoss()

    for epoch in range(epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0

        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = (
                input_stu_ids.to(device), 
                input_exer_ids.to(device), 
                input_knowledge_embs.to(device), 
                labels.to(device)
            )

            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0

        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_snapshot(net, MODEL_DIR / f"model_epoch{epoch + 1}.pt")
        print(f"Epoch {epoch+1}/{epoch_n} finished. Model saved.\n")

    print("Training done. Saving student embeddings...")

    # Save student embeddings
    stu_emb = net.student_emb.weight.data.cpu().numpy()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EMB_DIR / "student_emb.npy"
    np.save(str(out_path), stu_emb)
    print(f"Student embedding saved to {out_path}")


def save_snapshot(model, filename):
    torch.save(model.state_dict(), str(filename))


if __name__ == '__main__':
    # Run directly without command-line arguments

    train()
