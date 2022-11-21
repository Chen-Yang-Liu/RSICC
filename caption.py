#!/usr/bin/env python3

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
# from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import time
import cv2
import numpy
# import transformer, models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visual_feat0(args, img_A_feat,img_B_feat):
    img_A = imread(args.img_A)
    img_B = imread(args.img_B)
    dif_feat = numpy.absolute((img_B_feat[0,:1024,:,:] - img_A_feat[0,:1024,:,:]).mean(dim=0).cpu().numpy())
    img_A_feat = img_A_feat[0,:1024,:,:].mean(dim=0).cpu().numpy()
    img_B_feat = img_B_feat[0,:1024,:,:].mean(dim=0).cpu().numpy()

    for i in range(1):
        # print(i)
        headmap_A = img_A_feat[:,:]
        headmap_B = img_B_feat[:, :]


        headmap_A = dif_feat
        headmap_B = dif_feat

        headmap_A /= np.max(headmap_A)  # (7,7)
        headmap_B /= np.max(headmap_B)  # (7,7)

        headmap_A = cv2.resize(headmap_A, (img_A.shape[1], img_A.shape[0]))
        headmap_A = np.uint8(255 * headmap_A)
        headmap_A = cv2.applyColorMap(255-headmap_A, cv2.COLORMAP_JET)
        superimposed_img_A = headmap_A #* 0.2+ img_A * 0.8
        cv2.imwrite('F:/LCY/change_caption/Change_Captioning_Transformer/feat_image/resnet/A/'+str(i) + '.png', superimposed_img_A)

        headmap_B = cv2.resize(headmap_B, (img_A.shape[1], img_A.shape[0]))
        headmap_B = np.uint8(255 * headmap_B)
        headmap_B = cv2.applyColorMap(255-headmap_B, cv2.COLORMAP_JET)
        superimposed_img_B = headmap_B #* 0.2 + img_B * 0.8
        cv2.imwrite('F:/LCY/change_caption/Change_Captioning_Transformer/feat_image/resnet/B/' + str(i) + '.png', superimposed_img_B)


def visual_feat(args, encoder_out):
    img_A = imread(args.img_A)
    img_B = imread(args.img_B)
    dif_feat = numpy.absolute((encoder_out[:, 0, 512:,0].view(14,14,512) - encoder_out[:, 0, :512,0].view(14,14,512)).mean(dim=-1).cpu().numpy())
    img_A_feat = encoder_out[:, 0, :512,0].view(14,14,512).mean(dim=-1).cpu().numpy()
    img_B_feat = encoder_out[:, 0, 512:,0].view(14,14,512).mean(dim=-1).cpu().numpy()
    for i in range(1):
        # print(i)
        headmap_A = img_A_feat[:,:]
        headmap_B = img_B_feat[:, :]


        headmap_A = dif_feat
        headmap_B = dif_feat

        headmap_A /= np.max(headmap_A)  # (7,7)
        headmap_B /= np.max(headmap_B)  # (7,7)

        headmap_A = cv2.resize(headmap_A, (img_A.shape[1], img_A.shape[0]))
        headmap_A = np.uint8(255 * headmap_A)
        headmap_A = cv2.applyColorMap(255-headmap_A, cv2.COLORMAP_JET)
        superimposed_img_A = headmap_A #* 0.2+ img_A * 0.8
        cv2.imwrite('F:/LCY/change_caption/Change_Captioning_Transformer/feat_image/RSICCformer/A/'+str(i) + '.png', superimposed_img_A)

        headmap_B = cv2.resize(headmap_B, (img_A.shape[1], img_A.shape[0]))
        headmap_B = np.uint8(255 * headmap_B)
        headmap_B = cv2.applyColorMap(255-headmap_B, cv2.COLORMAP_JET)
        superimposed_img_B = headmap_B #* 0.2 + img_B * 0.8
        cv2.imwrite('F:/LCY/change_caption/Change_Captioning_Transformer/feat_image/RSICCformer/B/' + str(i) + '.png', superimposed_img_B)

def save_captions(args, word_map, hypotheses):
    result_json_file = {}
    reference_json_file = {}
    kkk = -1
    for item in hypotheses:
        kkk += 1
        line_hypo = ""

        for word_idx in item:
            word = get_key(word_map, word_idx)
            # print(word)
            line_hypo += word[0] + " "

        result_json_file[str(kkk)] = []
        result_json_file[str(kkk)].append(line_hypo)

        line_hypo += "\r\n"

    print(result_json_file)
    with open('eval_results/'+'/'+args.encoder_image + "_" +args.encoder_feat+"_" +args.decoder + '_res.json', 'w') as f:
        json.dump(result_json_file, f)



def get_key(dict_, value):
  return [k for k, v in dict_.items() if v == value]

def evaluate_transformer(args,encoder_image,encoder_feat,decoder):
    # Load model
    encoder_image = encoder_image.to(device)
    encoder_image.eval()
    encoder_feat = encoder_feat.to(device)
    encoder_feat.eval()
    decoder = decoder.to(device)
    decoder.eval()

    # Load word map (word2ix)
    word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    """
    Evaluation for decoder: transformer
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    beam_size = args.beam_size
    Caption_End = False
    # DataLoader

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    with torch.no_grad():

        k = beam_size

        # Read image and process
        img_A = imread(args.img_A)
        img_A = img_A.transpose(2, 0, 1)
        img_A = img_A / 255.
        img_A = torch.FloatTensor(img_A).to(device)
        img_B = imread(args.img_B)
        img_B = img_B.transpose(2, 0, 1)
        img_B = img_B / 255.
        img_B = torch.FloatTensor(img_B).to(device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([normalize])

        img_A = transform(img_A)  # (3, 256, 256)
        img_A = img_A.unsqueeze(0)  # (1, 3, 256, 256)
        img_B = transform(img_B)  # (3, 256, 256)
        img_B = img_B.unsqueeze(0)  # (1, 3, 256, 256)
        # Encode
        imgs_A = encoder_image(img_A)
        imgs_B = encoder_image(img_B)  # encoder_image :[1, 1024,14,14]
        # visual_feat0(args,imgs_A,imgs_B)

        encoder_out = encoder_feat(imgs_A, imgs_B) # encoder_out: (S, batch, feature_dim)

        # 可视化
        # visual_feat(args,encoder_out)



        tgt = torch.zeros(52, k).to(device).to(torch.int64)
        tgt_length = tgt.size(0)
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt[0, :] = torch.LongTensor([word_map['<start>']]*k).to(device) # k_prev_words:[52,k]
        # Tensor to store top k sequences; now they're just <start>
        seqs = torch.LongTensor([[word_map['<start>']]*1] * k).to(device)  # [1,k]
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)
        # Lists to store completed sequences and scores
        complete_seqs = []
        complete_seqs_scores = []
        step = 1

        k_prev_words = tgt.permute(1,0)
        S = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # # We'll treat the problem as having a batch size of k, where k is beam_size
        encoder_out = encoder_out.expand(S,k, encoder_dim)  # [S,k, encoder_dim]
        encoder_out = encoder_out.permute(1,0,2)

        # Start decoding
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            tgt = k_prev_words.permute(1,0)
            tgt_embedding = decoder.vocab_embedding(tgt)
            tgt_embedding = decoder.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

            encoder_out = encoder_out.permute(1, 0, 2)
            pred = decoder.transformer(tgt_embedding, encoder_out, tgt_mask=mask)  # (length, batch, feature_dim)
            encoder_out = encoder_out.permute(1, 0, 2)
            pred = decoder.wdc(pred)  # (length, batch, vocab_size)
            scores = pred.permute(1,0,2)  # (batch,length,  vocab_size)
            scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            # top_k_scores: [s, 1]
            scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)


            # Convert unrolled indices to actual indices of scores
            # prev_word_inds = top_k_words // vocab_size  # (s)
            # if max(top_k_words)>vocab_size:
            #     print(">>>>>>>>>>>>>>>>>>")
            prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            # Set aside complete sequences
            if len(complete_inds) > 0:
                Caption_End = True
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            # Important: this will not work, since decoder has self-attention
            # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
            k_prev_words = k_prev_words[incomplete_inds]
            k_prev_words[:, :step + 1] = seqs  # [s, 52]
            # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1


        # choose the caption which has the best_score.
        if (len(complete_seqs_scores) > 0):
            assert Caption_End
            indices = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[indices]

            # References
            # img_caps = allcaps[0].tolist()
            # img_captions = list(
            #     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
            #         img_caps))  # remove <start> and pads
            # references.append(img_captions)

            # Hypotheses
            # tmp_hyp = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
            # assert len(references) == len(hypotheses)
    # captions
    save_captions(args, word_map, hypotheses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change_Captioning')
    parser.add_argument('--img_A',  default='./Example/A/train_000016.png')
    parser.add_argument('--img_B',  default='./Example/B/train_000016.png')
    parser.add_argument('--data_folder', default="./data/",help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="Levir_CC_5_cap_per_img_5_min_word_freq",help='base name shared by data files.')

    parser.add_argument('--encoder_image', default="resnet101") # inception_v3 or vgg16 or vgg19 or resnet50 or resnet101 or resnet152
    parser.add_argument('--encoder_feat', default="MCCFormers_diff_as_Q") # MCCFormers-S or MCCFormers-D
    parser.add_argument('--decoder', default="trans", help="decoder img2txt")  #
    parser.add_argument('--Split', default="TEST", help='which')
    parser.add_argument('--epoch', default="epoch", help='which')
    parser.add_argument('--beam_size', type=int, default=1, help='beam_size.')
    parser.add_argument('--path', default="./models_checkpoint/", help='model checkpoint.') #  ./models_checkpoint/data/2-times/RSICCformer_D/Simis_baseline/
    args = parser.parse_args()


    filename = os.listdir(args.path)
    for i in range(len(filename)):
        if (args.epoch in filename[i]) or (args.encoder_feat not in filename[i])or (args.decoder not in filename[i]) :
            continue
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

        checkpoint_path = os.path.join(args.path, filename[i])
        print(args.path + filename[i])

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        encoder_image = checkpoint['encoder_image']
        encoder_feat = checkpoint['encoder_feat']
        decoder = checkpoint['decoder']

        evaluate_transformer(args,encoder_image,encoder_feat,decoder)

        time.sleep(10)