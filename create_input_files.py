from utils import create_input_files
import time

if __name__ == '__main__':

    print('create_input_files START at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))

    create_input_files(dataset='LEVIR_CC',
                       karpathy_json_path=r'./Levir_CC_dataset/LevirCCcaptions.json',
                       image_folder=r'./Levir_CC_dataset/images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder=r'./data',
                       max_len=50)

    print('create_input_files END at: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))