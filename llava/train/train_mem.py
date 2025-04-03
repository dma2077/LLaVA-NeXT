from llava.train.train import train

if __name__ == "__main__":
    import wandb
    wandb.login(key="f3b76ea66a38b2a211dc706fa95b02c761994b73")
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    # import debugpy
    # debugpy.listen(('0.0.0.0', 4123))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    train()
