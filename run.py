from NewFinalUtils import ARG
from AnswerSelectMethods import ConcatAnswerSelectMethod

import logging

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )   

# 先创建一个参数集
args = ARG(per_gpu_train_batch_size=2,
            gradient_accumulation_steps = 8,
            device='cuda:1',
			model_name="./checkpoint-3285",
			tokenizer_name="/data/package/chinese_roberta_wwm_large_ext_pytorch",
            topk=2,
            num_train_epochs = 5,
            save_dir = './23_4_top2_on3285_models')

method = ConcatAnswerSelectMethod(args=args)

method.read_all_files(NBEST_FILE_PATH='23_4_cmrc1_drcd1_cail1_lic_train_lic_nbest_predictions_utf8.json',
                      DATA_FILE_PATH='/data/bzw/MRC/data/lic2020/dureader_robust-data/train.json')

method.build_train_data()
# 在这里，loss不降
method.fit()
