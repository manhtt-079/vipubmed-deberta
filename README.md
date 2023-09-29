# ViPubmedDeBERTa: A Pre-trained Model for Vietnamese Biomedical Text


## Model description
The architecture of ViPubmedDeBERTa relies on ViDeBERTa [(Tran et al., 2023)](https://aclanthology.org/2023.findings-eacl.79.pdf). ViDeBERTa, a recently developed pre- trained monolingual language model for the Vietnamese language, leverages the extensive CC100 dataset, which composes 138GB of uncompressed texts derived from web crawls of monolingual data sources [(Conneau et al., 2019)](https://arxiv.org/abs/1911.02116). ViDeBERTa is built upon the architecture of DeBERTaV3 [(He et al., 2021)](https://arxiv.org/abs/2111.09543). The model undergoes training using selfsupervised learning objectives, specifically Masked Language Modeling (MLM) and Relationaware Token Discrimination (RTD), aiming to optimize its performance. Furthermore, a novel approach known as Gradient Disentangled Embedding Sharing (GDES), which employs weight sharing techniques, is incorporated to further enhance the overall efficacy of the model.

## Model variations
- [vipubmed-deberta-xsmall](https://huggingface.co/manhtt-079/vipubmed-deberta-xsmall): 22M backbone parameters
- [vipubmed-deberta-base](https://huggingface.co/manhtt-079/vipubmed-deberta-base): 86M backbone parameters

## How to use
You can use this model directly with a pipeline for masked language modeling:<br>
**_NOTE:_**  The input text should be already word-segmented, you can use [Pyvi](https://github.com/trungtv/pyvi) (Python Vietnamese Core NLP Toolkit) to segment word before passing to the model.
```python
>>> from transformers import pipeline
>>> model = pipeline('fill-mask', model='manhtt-079/vipubmed-deberta-base')
>>> text_with_mask = """Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS ) . FDCS là bệnh rất hiếm ảnh_hưởng đến tế_bào trình_diện kháng_nguyên đuôi gai và thường bị chẩn_đoán nhầm . Phẫu_thuật được coi là phương_thức điều_trị tốt nhất , tiếp_theo là hóa_trị . Trong trường_hợp của chúng_tôi , [MASK] cắt bỏ không_thể thực_hiện được , do đó bệnh_nhân được hóa_trị hai dòng , sau đó là cấy_ghép tủy xương , sau đó là hóa_trị ba với đáp_ứng trao_đổi chất hoàn_toàn được thấy trên"""
>>> model(text_with_mask)

[{'score': 0.8480948805809021,
  'token': 1621,
  'token_str': 'phẫu_thuật',
  'sequence': 'Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS ). FDCS là bệnh rất hiếm ảnh_hưởng đến tế_bào trình_diện kháng_nguyên đuôi gai và thường bị chẩn_đoán nhầm. Phẫu_thuật được coi là phương_thức điều_trị tốt nhất, tiếp_theo là hóa_trị. Trong trường_hợp của chúng_tôi, phẫu_thuật cắt bỏ không_thể thực_hiện được, do đó bệnh_nhân được hóa_trị hai dòng, sau đó là cấy_ghép tủy xương, sau đó là hóa_trị ba với đáp_ứng trao_đổi chất hoàn_toàn được thấy trên'},
 {'score': 0.1136574074625969,
  'token': 83,
  'token_str': 'việc',
  'sequence': 'Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS ). FDCS là bệnh rất hiếm ảnh_hưởng đến tế_bào trình_diện kháng_nguyên đuôi gai và thường bị chẩn_đoán nhầm. Phẫu_thuật được coi là phương_thức điều_trị tốt nhất, tiếp_theo là hóa_trị. Trong trường_hợp của chúng_tôi, việc cắt bỏ không_thể thực_hiện được, do đó bệnh_nhân được hóa_trị hai dòng, sau đó là cấy_ghép tủy xương, sau đó là hóa_trị ba với đáp_ứng trao_đổi chất hoàn_toàn được thấy trên'},
 {'score': 0.014141257852315903,
  'token': 589,
  'token_str': 'phương_pháp',
  'sequence': 'Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS ). FDCS là bệnh rất hiếm ảnh_hưởng đến tế_bào trình_diện kháng_nguyên đuôi gai và thường bị chẩn_đoán nhầm. Phẫu_thuật được coi là phương_thức điều_trị tốt nhất, tiếp_theo là hóa_trị. Trong trường_hợp của chúng_tôi, phương_pháp cắt bỏ không_thể thực_hiện được, do đó bệnh_nhân được hóa_trị hai dòng, sau đó là cấy_ghép tủy xương, sau đó là hóa_trị ba với đáp_ứng trao_đổi chất hoàn_toàn được thấy trên'},
 {'score': 0.0024715897161513567,
  'token': 454,
  'token_str': 'điều_trị',
  'sequence': 'Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS ). FDCS là bệnh rất hiếm ảnh_hưởng đến tế_bào trình_diện kháng_nguyên đuôi gai và thường bị chẩn_đoán nhầm. Phẫu_thuật được coi là phương_thức điều_trị tốt nhất, tiếp_theo là hóa_trị. Trong trường_hợp của chúng_tôi, điều_trị cắt bỏ không_thể thực_hiện được, do đó bệnh_nhân được hóa_trị hai dòng, sau đó là cấy_ghép tủy xương, sau đó là hóa_trị ba với đáp_ứng trao_đổi chất hoàn_toàn được thấy trên'},
 {'score': 0.002370780799537897,
  'token': 485,
  'token_str': 'quá_trình',
  'sequence': 'Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS ). FDCS là bệnh rất hiếm ảnh_hưởng đến tế_bào trình_diện kháng_nguyên đuôi gai và thường bị chẩn_đoán nhầm. Phẫu_thuật được coi là phương_thức điều_trị tốt nhất, tiếp_theo là hóa_trị. Trong trường_hợp của chúng_tôi, quá_trình cắt bỏ không_thể thực_hiện được, do đó bệnh_nhân được hóa_trị hai dòng, sau đó là cấy_ghép tủy xương, sau đó là hóa_trị ba với đáp_ứng trao_đổi chất hoàn_toàn được thấy trên'}]
```

#### Get features:
- With PyTorch:
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('manhtt-079/vipubmed-deberta-base')
model = AutoModel.from_pretrained("manhtt-079/vipubmed-deberta-base")
text = "Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS )."
model_inputs = tokenizer(text, return_tensors='pt')
outputs = model(**model_inputs)
```

- With TensorFlow
```python
from transformers import AutoTokenizer, TFAutoModel
tokenizer = AutoTokenizer.from_pretrained('manhtt-079/vipubmed-deberta-base')
model = TFAutoModel.from_pretrained("manhtt-079/vipubmed-deberta-base")
text = "Chúng_tôi mô_tả một trường_hợp bệnh_nhân nữ 44 tuổi được chẩn_đoán sarcoma tế_bào tua nang ( FDCS )."
model_inputs = tokenizer(text, return_tensors='tf')
outputs = model(**model_inputs)
```

## Pre-training data
The ViPubMedDeBERTa model was pre-trained on [ViPubmed](https://github.com/vietai/ViPubmed), a dataset consisting of 20M Vietnamese Biomedical abstracts generated by large scale translation.

## Training procedure
### Data deduplication
A fuzzy deduplication, targeting documents with high overlap, was conducted at the document level to enhance quality and address overfitting. Employing Locality Sensitive Hashing (LSH) with a threshold of 0.9 ensured the removal of documents with overlap exceeding 90%. This process resulted in an average reduction of the dataset's size by 3%.
### Pretraining
We employ our model based on the [ViDeBERTa](https://github.com/HySonLab/ViDeBERTa) architecture and leverage its pre-trained checkpoint to continue pre-training. Our model was trained on a single A100 GPU (40GB) for 350 thousand steps, with a batch size of 16 and gradient accumulation steps set to 4 (resulting in a total of 64). The sequence length was limited to 512 tokens and the model peak learning rate of 1e-4.

## Evaluation results