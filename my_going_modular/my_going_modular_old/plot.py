import matplotlib.pyplot as plt


def plot_results(results, title=None):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results['train_loss'], label='train')
    plt.plot(results['test_loss'],  label='test')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results['train_acc'], label='train')
    plt.plot(results['test_acc'], label='test')
    plt.title('Acciracy')
    plt.legend()
