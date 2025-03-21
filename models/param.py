import argparse

class Config:
    def __init__(self, batch_size=16, lr=0.001, epochs=200, alpha = 0.6, resume_epoch=0):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.resume_epoch = resume_epoch
    
        self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Training Script")
        parser.add_argument("--batch_size", type=int, default=self.batch_size, help="Batch size for training")
        parser.add_argument("--lr", type=float, default=self.lr, help="Learning rate")
        parser.add_argument("--epochs", type=int, default=self.epochs, help="Number of training epochs")
        parser.add_argument("--resume_epoch", type=int, default=self.resume_epoch, help="Epoch to resume training from")
        args = parser.parse_args()
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.resume_epoch = args.resume_epoch


    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(f"Config has no attribute {k}")
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)
    
# def main():
#     config = Config()
#     print(config)
#     arg = config.__dict__

#     config.update(batch_size=32, lr=0.01)
#     print(config)

#     config.update(batch_size=64, lr=0.001, epochs=100)
#     print(config)

#     print(arg['batch_size'])

# if __name__ == "__main__":
#     main()