import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# loss_file = open('/Users/markus/workspace/master/Master/GAN/GAN_log/2017-04-26_ImgCapFalse_WordEmbedding.WORD2VEC_Vocab1000_Seq10_Batch128_EmbSize50_NoiseMode.REPEAT_Noise50_PreInitPreInit.NONE_Dataset_all_flowers_500hidden_dropout0.2/loss.txt', 'r')
# loss_fix_file = open('/Users/markus/workspace/master/Master/GAN/GAN_log/2017-04-26_ImgCapFalse_WordEmbedding.WORD2VEC_Vocab1000_Seq10_Batch128_EmbSize50_NoiseMode.REPEAT_Noise50_PreInitPreInit.NONE_Dataset_all_flowers_500hidden_dropout0.2/loss-fix.txt', 'w+')
# loss_lines = loss_file.readlines()
# loss_fix_file.writelines(loss_lines[1::2])
# loss_file.close()
# loss_fix_file.close()


log_folder = 'GAN/GAN_log/'

model_1 = '2017-05-12_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers'
model_2 = '2017-05-12_ImgCapFalse_onehot_Vocab1000_Seq12_Batch64_EmbSize50_repeat_Noise50_PreInitNone_Dataset_10_all_flowers_0.75dropout'

data_1 = np.genfromtxt(
	log_folder + model_1 + "/loss.txt",
	delimiter=',',
	skip_header=1,
	skip_footer=17,
	names=['epoch', 'batch', 'g_loss', 'g_acc', 'd_loss_gen', 'd_acc_gen', 'd_loss_train', 'd_acc_train'])

data_2 = np.genfromtxt(
	log_folder + model_2 + "/loss.txt",
	delimiter=',',
	skip_header=1,
	skip_footer=0,
	names=['epoch', 'batch', 'g_loss', 'g_acc', 'd_loss_gen', 'd_acc_gen', 'd_loss_train', 'd_acc_train'])

fig = plt.figure()

diagram = fig.add_subplot(111)

d_loss_train_1 = data_1["d_loss_train"]
d_loss_gen_1 = data_1["d_loss_gen"]
d_loss_1 = (d_loss_gen_1 + d_loss_train_1) / 2

d_loss_train_2 = data_2["d_loss_train"]
d_loss_gen_2 = data_2["d_loss_gen"]
d_loss_2 = (d_loss_gen_2 + d_loss_train_2) / 2

# ax1.set_title("Accuracy - 15 seqLength - Two Flowers")
diagram.set_xlabel('Epoch')
diagram.set_ylabel('Loss')

colors = ['#F95400', '#004FA2', '#F9C000', 'y']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = markers + [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']

skip = 10
first = None
# ax1.plot(data['epoch'][:first:skip], data['d_acc_gen'][:first:skip], c='b', label='discriminator_fake_accuracy')
# ax1.plot(data['epoch'][:first:skip], data['d_acc_train'][:first:skip], c='g', label='discriminator_real_accuracy')
# ax1.plot(data['epoch'][:first:skip], data['g_acc'][:first:skip], c='r', label='generator_accuracy')

# ax1.plot(data['epoch'][:first:skip], data['d_loss_gen'][:first:skip], c='b', label='discriminator_fake_loss')
# ax1.plot(data['epoch'][:first:skip], data['d_loss_train'][:first:skip], c='g', label='discriminator_real_loss')

diagram.plot(data_1['epoch'][:first:skip], data_1['g_loss'][:first:skip], c=colors[0], label='Generator')
diagram.plot(data_1['epoch'][:first:skip], d_loss_1[:first:skip], c=colors[1], label='Discriminator')

diagram.plot(data_1['epoch'][:first:skip], data_2['g_loss'][:first:skip], c=colors[0], linestyle=':', label='Generator (Dropout)')
diagram.plot(data_1['epoch'][:first:skip], d_loss_2[:first:skip], c=colors[1], linestyle=':', label='Discriminator (Dropout)')

leg = diagram.legend()

plt.show()
