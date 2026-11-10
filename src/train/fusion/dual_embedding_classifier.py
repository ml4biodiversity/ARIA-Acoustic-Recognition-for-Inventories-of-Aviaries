import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

class DualEmbeddingClassifier(nn.Module):
    """
    Fusion classifier combining BirdNET and PERCH embeddings.
    Enhanced with batch normalization and improved architecture.
    """
    
    def __init__(self, n_species, birdnet_dim=93, perch_dim=87,
                 hidden_dims=[1024, 512, 256], dropout=0.3):
        """
        Args:
            n_species: Number of bird species to classify
            birdnet_dim: Dimension of BirdNET embeddings
            perch_dim: Dimension of PERCH embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_species = n_species
        self.birdnet_dim = birdnet_dim
        self.perch_dim = perch_dim
        
        # Project BirdNET embeddings
        self.birdnet_projection = nn.Sequential(
            nn.Linear(birdnet_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Project PERCH embeddings
        self.perch_projection = nn.Sequential(
            nn.Linear(perch_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Build fusion layers
        layers = []
        input_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.fusion_layers = nn.Sequential(*layers)
        
        # Output layer
        self.classifier = nn.Linear(hidden_dims[-1], n_species)
    
    def forward(self, combined_embeddings):
        """
        Forward pass through the network.
        
        Args:
            combined_embeddings: (batch_size, birdnet_dim + perch_dim)
        
        Returns:
            logits: (batch_size, n_species)
        """
        # Split combined embeddings
        birdnet_emb = combined_embeddings[:, :self.birdnet_dim]
        perch_emb = combined_embeddings[:, self.birdnet_dim:]
        
        # Project each embedding type
        birdnet_proj = self.birdnet_projection(birdnet_emb)
        perch_proj = self.perch_projection(perch_emb)
        
        # Concatenate projected embeddings
        fused = torch.cat([birdnet_proj, perch_proj], dim=1)
        
        # Pass through fusion layers
        features = self.fusion_layers(fused)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, combined_embeddings):
        """
        Make predictions with confidence scores.
        
        Args:
            combined_embeddings: (batch_size, birdnet_dim + perch_dim)
        
        Returns:
            predictions: (batch_size,) predicted class indices
            confidences: (batch_size,) confidence scores
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(combined_embeddings)
            probabilities = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
        
        return predictions, confidences


class BirdEmbeddingDataset(Dataset):
    """Dataset for bird embeddings"""
    
    def __init__(self, embeddings, labels, is_rare):
        """
        Args:
            embeddings: (n_samples, embedding_dim) array
            labels: (n_samples,) array of class labels
            is_rare: (n_samples,) boolean array indicating rare species
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
        self.is_rare = torch.FloatTensor(is_rare)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.is_rare[idx]


class DualEmbeddingTrainer:
    """Trainer for dual embedding classifier"""
    
    def __init__(self, model, learning_rate=0.0005, weight_decay=1e-4,
                 rare_species_weight=1.2, use_focal_loss=True):
        """
        Initialize trainer.
        
        Args:
            model: DualEmbeddingClassifier instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            rare_species_weight: Weight multiplier for rare species
            use_focal_loss: Whether to use focal loss
        """
        self.model = model
        self.rare_species_weight = rare_species_weight
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler (Cosine Annealing with Warm Restarts)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6
        )
        
        # Initialize loss function
        if use_focal_loss:
            try:
                from losses import FocalLoss
                self.criterion = FocalLoss(
                    gamma=2.0,
                    label_smoothing=0.1
                )
                print(f"✓ Using Focal Loss")
            except ImportError:
                print(f"⚠️  FocalLoss not found, using CrossEntropyLoss")
                self.criterion = nn.CrossEntropyLoss(
                    reduction='mean',
                    label_smoothing=0.1
                )
        else:
            self.criterion = nn.CrossEntropyLoss(
                reduction='mean',
                label_smoothing=0.1
            )
        
        print(f"✓ Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Criterion: {type(self.criterion).__name__}")
    
    def compute_class_weights(self, labels):
        """
        Compute balanced class weights for imbalanced dataset.
        
        Args:
            labels: Array of class labels
        
        Returns:
            Tensor of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        
        return torch.FloatTensor(weights)
    
    def train_epoch(self, train_loader, class_weights=None):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            class_weights: Optional class weights
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for embeddings, labels, is_rare in train_loader:
            # Move to device
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            is_rare = is_rare.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(embeddings)
            
            # Compute loss
            if isinstance(self.criterion, nn.CrossEntropyLoss) and class_weights is not None:
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights.to(self.device),
                    reduction='mean',
                    label_smoothing=0.1
                )
                loss = criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, val_loader):
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels, is_rare in val_loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(embeddings)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Ensure scalar
                if loss.dim() > 0:
                    loss = loss.mean()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def train(self, train_loader, val_loader, n_epochs, class_weights=None,
              checkpoint_path=None, start_epoch=0):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of epochs
            class_weights: Optional class weights
            checkpoint_path: Path to save checkpoints
            start_epoch: Starting epoch (for resuming)
        
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 25
        
        print(f"\nStarting training from epoch {start_epoch + 1}")
        print("="*60)
        
        for epoch in range(start_epoch, n_epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader, class_weights)
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Check for improvement
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                if checkpoint_path:
                    best_path = checkpoint_path.replace('_checkpoint.pth', '_best.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_metrics['accuracy'],
                        'val_loss': val_metrics['loss']
                    }, best_path)
                    print(f"  ✓ New best model (acc: {best_val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            if checkpoint_path and (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'history': history
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
            
            print()
        
        print("="*60)
        print("Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return history