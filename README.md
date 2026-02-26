# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Develop an image classification model using transfer learning with VGG19 architecture for the given dataset.


## Neural Network Model
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/e4c61cf3-314f-4b2f-9833-dd2cb77b795b" />


## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.


### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 
Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.


### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.



## PROGRAM

### Name: VENKATESAN R

### Register Number: 212224230299

```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models.vgg import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
rom torchsummary import summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
  train_losses=[]
  val_losses=[]
  model.train()
  for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss/len(train_loader))


    # Compute validation loss
    model.eval()
    val_loss=0.0
    with torch.no_grad():
      for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss=criterion(outputs,labels.unsqueeze(1).float())
        val_loss+=loss.item()
    val_losses.append(val_loss/len(test_loader))
    model.train()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: VENKATESAN R")
    print("Register Number: 212224230299")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot


<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/3ef7710c-0e13-43ad-837a-1ce8e4e1fb75" />


## Confusion Matrix


<img width="756" height="659" alt="{2C6A36AC-3F86-4375-B735-6BE289DBA2B4}" src="https://github.com/user-attachments/assets/3fe8d28e-dcc4-4cee-8f58-3e200abada4b" />


## Classification Report

<img width="586" height="241" alt="{CE6A8DA7-F076-49FF-A7A3-EDF74A860838}" src="https://github.com/user-attachments/assets/7adc3187-a8e7-47f3-adae-62a6c668c283" />


### New Sample Data Prediction

<img width="389" height="450" alt="{D4F706DB-B97A-48B8-B002-7B31BA0FCC2E}" src="https://github.com/user-attachments/assets/afc21856-5e27-401d-8b0f-bb284f4d7b3d" />

<img width="380" height="444" alt="{36A77626-97CC-4F58-9566-33D1D6B9C062}" src="https://github.com/user-attachments/assets/1fdd4a3b-2d05-4d6f-b878-2a77006a8bcb" />



## RESULT
Developing a Neural Network Classification Model using Transfer Learning was Successfully built
