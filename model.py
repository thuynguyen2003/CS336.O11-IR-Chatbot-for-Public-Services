from rank_bm25 import BM25Okapi, BM25Plus
import re
import numpy as np
from underthesea import text_normalize
import pandas as pd
from pyvi import ViTokenizer
import heapq
import torch
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import chromadb
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from Levenshtein import ratio as lev
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# preprocessing function
def chuan_hoa_unicode_go_dau(text):
  return text_normalize(text)

def viet_thuong(text):
	return text.lower()

def chuan_hoa_dau_cau(text):
	text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text

def chuan_hoa_cau(doc):
    pattern = r'(\w)([^\s\w])'
    result1 = re.sub(pattern, r'\1 \2', doc)

    pattern = r'([^\s\w])(\w)'
    result2 = re.sub(pattern, r'\1 \2', result1)

    pattern = r'\s+'
    # Loại bỏ khoảng trắng thừa
    result = re.sub(pattern, ' ', result2)
    return result

def my_pre_processing(doc):
  doc = chuan_hoa_unicode_go_dau(doc)
  doc = chuan_hoa_dau_cau(doc)
  doc = chuan_hoa_cau(doc)
  doc = viet_thuong(doc)
  return doc

# Phân loại in-out
model_cls_in_out = joblib.load('./source/cls_in_out_svm.joblib')
vectorizer_in_out = joblib.load('./source/cls_in_out_tfidf_vectorizer.joblib')
def Cls_in_out(query, model, vectorizer):
  # tiền xử lý + tách từ
  query = my_pre_processing(query)
  if query == "xin chào":
    return 0
  query_tokenized = ViTokenizer.tokenize(query)

  # vector hóa
  vector = vectorizer.transform([query_tokenized])
  # predict
  result = model.predict(vector)[0]

  return result

# Thêm dấu
corrector = pipeline("text2text-generation", model="ThuyNT03/CS336_bartpho-syllable-base_add-accent")

def cosine_similarity_score(sentence_1, sentence_2):
    # Tạo vectorizer
    vectorizer = CountVectorizer()

    # Huấn luyện vectorizer với câu đầu vào
    vectorizer.fit([sentence_1, sentence_2])

    # Biến đổi câu thành vector BoW
    vector_1 = vectorizer.transform([sentence_1]).toarray()
    vector_2 = vectorizer.transform([sentence_2]).toarray()

    # Tính cosine similarity
    similarity = cosine_similarity(vector_1, vector_2)[0][0]

    return similarity

def top_n_indexes(lst, n):
    top_items = heapq.nlargest(n, enumerate(lst), key=lambda x: x[1])
    return [i for i, s in top_items]

def BM25_retrieval(query, seg_question_corpus, top_BM25):
  query = my_pre_processing(query)
  word_tokenized_query = ViTokenizer.tokenize(query).split(" ")
  # xử lý ở level word với question
  tokenized_word_question_corpus = [doc.split(" ") for doc in seg_question_corpus]
  bm25_word_question = BM25Plus(tokenized_word_question_corpus)
  word_score_question = bm25_word_question.get_scores(word_tokenized_query)
  BM25_result = top_n_indexes(word_score_question, n=top_BM25)
  return BM25_result

def SimCSE_retrieval(query, SimCSE_set, top_Sim):
  from sentence_transformers import CrossEncoder
  query = my_pre_processing(query)
  # Sim_CSE_model_question = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
  # Sim_CSE_word_ques_embeddings = torch.load('./source/word_ques_embeddings.pth')
  Sim_CSE_model_question = SimCSE_set[0]
  Sim_CSE_word_ques_embeddings = SimCSE_set[1]

  seg_query = ViTokenizer.tokenize(query)
  query_vector = Sim_CSE_model_question.encode(seg_query)
  SimCSE_word_scores = list(cosine_similarity([query_vector], Sim_CSE_word_ques_embeddings)[0])
  SimCSE_result = top_n_indexes(SimCSE_word_scores, n=top_Sim)
  return SimCSE_result

def Para_retriveval(query, para_set, top_para):
  query = my_pre_processing(query)
  from sentence_transformers import SentenceTransformer, CrossEncoder
  import torch
  retri_model = para_set[0]
  para_question_embeddings = para_set[1]

  query_embed = retri_model.encode([query], device = device)
  para_score = cosine_similarity(query_embed, para_question_embeddings)[0]
  Para_result = top_n_indexes(para_score, n = top_para)
  return Para_result

def Rerank(query, retrieval_result, question_corpus, reranker, top_n):
  #rerank_model_name = 'unicamp-dl/mMiniLM-L6-v2-mmarco-v2'
  query = my_pre_processing(query)
  #reranker = CrossEncoder(rerank_model_name)
  scores = reranker.predict([(query, question_corpus[i]) for i in retrieval_result])
  id_score = list(zip(retrieval_result, scores))
  sorted_id_score = sorted(id_score, key=lambda x: x[1], reverse=True)[:(min(len(retrieval_result), top_n))]
  return sorted_id_score

def retrieval(query, question_corpus, seg_question_corpus, models, top_n = 15, thread_hold = 0.2, rerank = True):
  BM25_result = BM25_retrieval(query, seg_question_corpus, top_n)
  SimCSE_result = SimCSE_retrieval(query, models['Sim_CSE'], top_n)
  Para_result = Para_retriveval(query, models['para'], top_n)
  retrieval_result = list(set(BM25_result + SimCSE_result + Para_result))
  #sents_retri = [question_corpus[i] for i in retrieval_result]

  scores_filter = []
  #while len(scores_filter) == 0 and thread_hold >= 0:
      #scores_filter = []
  for id in retrieval_result:
      score = cosine_similarity_score(my_pre_processing(query), question_corpus[id])
      if score >= thread_hold:
          #print(score)
          scores_filter.append((score, id))
      #thread_hold -= 0.1
  scores_filter = sorted(scores_filter, key = lambda x : x[0], reverse=True)
  sent_filter = [i[1] for i in scores_filter]

  if rerank == False:
    return scores_filter
  #print("test")
  rerank_result = Rerank(query, sent_filter, question_corpus, models['rerank'], top_n)
  sent_rerank = [i[0] for i in rerank_result]
  sent_rerank.append(-1)

  score_rerank = [i[1] for i in rerank_result]
  score_rerank = [(i - min(score_rerank))/(max(score_rerank) - min(score_rerank)) for i in score_rerank]
  data_rerank = {}
  for i in sent_rerank:
      data_rerank[i] = []

  for idx, id in enumerate(sent_rerank):
      for j in range(idx + 1, len(sent_rerank)):
          if id == -1:
              sent1 = my_pre_processing(query)
          else:
              sent1 = question_corpus[id]

          if sent_rerank[j] == -1:
              sent2 = my_pre_processing(query)
          else:
              sent2 = question_corpus[sent_rerank[j]]

          score = cosine_similarity_score(sent1, sent2) * score_rerank[idx]
          data_rerank[id].append(score)
          data_rerank[sent_rerank[j]].append(score)

  del data_rerank[-1]
  data_rerank = {key: sum(data)/len(data) for key, data in data_rerank.items()}
  scores_rerank = [{'corpus_id': key, 'score': score} for key, score in sorted(data_rerank.items(), key = lambda x: x[1], reverse = True)]

  return scores_rerank

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
df = pd.read_csv('./source/corpus.csv')
question_corpus = list(df['Question'])
seg_question_corpus = list(df['Seg_question'])
Sim_CSE_model = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
Sim_CSE_word_ques_embeddings = torch.load('./source/word_ques_embeddings.pth')

para_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
para_question_embeddings = torch.load('./source/para_embeddings.pth')

rerank_model = CrossEncoder('unicamp-dl/mMiniLM-L6-v2-mmarco-v2')

models = {'rerank': rerank_model, 'para': [para_model, para_question_embeddings], 'Sim_CSE': [Sim_CSE_model, Sim_CSE_word_ques_embeddings]}

def create_template():
  # PART ONE: SYSTEM
  system_template="Bạn là một Chatbot hỗ trợ tư vấn các thông tin liên quan đến các thủ tục thực hiện dịch vụ công. Vui lời trả lời ngắn gọn, dễ hiểu, dễ thực hiện."
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

  # PART TWO: Few Shot
  example_input = """
  Câu hỏi của người dùng: quy trình đổi chứng minh nhân dân sang căn cước công dân
  Tên của thủ tục hành chính: Đổi thẻ Căn cước công dân (thực hiện tại cấp trung ương)
  Ngữ cảnh: Nội dung của thủ tục hành chính:
  - Bước 1: Công dân đến địa điểm làm thủ tục cấp Căn cước công dân của Trung tâm dữ liệu quốc gia về dân cư, Cục Cảnh sát quản lý hành chính về trật tự xã hội, Bộ Công an hoặc thông qua Cổng dịch vụ công quốc gia, Cổng dịch vụ công Bộ Công an để đăng ký thời gian, địa điểm làm thủ tục đề nghị đổi thẻ Căn cước công dân.
  Trường hợp công dân không đủ điều kiện đổi thẻ Căn cước công dân thì từ chối tiếp nhận và nêu rõ lý do. Trường hợp công dân đủ điều kiện đổi thẻ Căn cước công dân thì thực hiện các bước sau:
  - Bước 2: Cán bộ thu nhận thông tin công dân tìm kiếm thông tin trong Cơ sở dữ liệu quốc gia về dân cư để lập hồ sơ đổi thẻ Căn cước công dân.
  + Trường hợp thông tin công dân không có sự thay đổi, điều chỉnh thì sử dụng thông tin của công dân trong cơ sở dữ liệu quốc gia về dân cư để lập hồ sơ đổi thẻ Căn cước công dân.
  + Trường hợp thông tin công dân có sự thay đổi, điều chỉnh thì đề nghị công dân xuất trình giấy tờ pháp lý chứng minh nội dung thay đổi để cập nhật, bổ sung thông tin trong hồ sơ đổi thẻ Căn cước công dân.
  - Bước 3: Tiến hành thu nhận vân tay, chụp ảnh chân dung của công dân.
  - Bước 4: In Phiếu thu nhận thông tin Căn cước công dân chuyển cho công dân kiểm tra, ký xác nhận; in Phiếu cập nhật, chỉnh sửa thông tin dân cư (nếu có) cho công dân kiểm tra, ký xác nhận.
  - Bước 5: Thu Căn cước công dân cũ, thu lệ phí (nếu có) và cấp giấy hẹn trả thẻ Căn cước công dân cho công dân (mẫu CC03 ban hành kèm theo Thông tư số 66/2015/TT-BCA ngày 15/12/2015 của Bộ trưởng Bộ Công an).
  - Thời gian tiếp nhận hồ sơ và thời gian trả kết quả: Từ thứ 2 đến thứ 6 hàng tuần (trừ ngày lễ, tết).
  - Bước 6: Nhận kết quả trực tiếp tại cơ quan Công an nơi tiếp nhận hồ sơ hoặc trả qua đường chuyển phát đến địa chỉ theo yêu cầu.
  """
  example_input_prompt = HumanMessagePromptTemplate.from_template(example_input)

  example_output = f"Quy trình đổi chứng minh nhân dân sang căn cước công dân được thực hiện như sau:\n- Bước 1: Đăng ký thời gian, địa điểm làm thủ tục đề nghị đổi thẻ Căn cước công dân.\n- Bước 2: Lập hồ sơ đổi thẻ Căn cước công dân dựa trên thông tin trong cơ sở dữ liệu quốc gia về dân cư.\n- Bước 3: Thu nhận vân tay và chụp ảnh chân dung của công dân.\n- Bước 4: In Phiếu thu nhận thông tin Căn cước công dân và Phiếu cập nhật, chỉnh sửa thông tin dân cư (nếu có) cho công dân kiểm tra, ký xác nhận.\n- Bước 5: Thu Căn cước công dân cũ, thu lệ phí (nếu có) và cấp giấy hẹn trả thẻ Căn cước công dân cho công dân.\n- Bước 6: Nhận kết quả trực tiếp tại cơ quan Công an nơi tiếp nhận hồ sơ hoặc trả qua đường chuyển phát đến địa chỉ theo yêu cầu."
  example_output_answer = AIMessagePromptTemplate.from_template(example_output)

  # PART THREE: HUMAN REQUEST
  human_template= "Câu hỏi của người dùng: {user_question}\nTên của thủ tục hành chính: {procedure_name}\nNgữ cảnh: Nội dung của thủ tục hành chính:\n{procedure}"
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  # PART THREE: COMPILE TO CHAT
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_input_prompt, example_output_answer, human_message_prompt])
  return chat_prompt

source_corpus = pd.read_csv("./source/Source_corpus.csv")
chat_prompt = create_template()
chat = ChatOpenAI(openai_api_key="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")
chat_plus = ChatOpenAI(openai_api_key="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", model='gpt-3.5-turbo-1106')
def Langchain_RAG(query):
  answer = {'query': query}
  retri_result = retrieval(query, question_corpus, seg_question_corpus, models)
  if len(retri_result) == 0:
    answer['answer'] = "Không tìm thấy thủ tục hành chính phù hợp"
    return answer
  corpus_id = retri_result[0]['corpus_id']
  #corpus_id = retri_result[0][1]
  #print(corpus_id)
  info = source_corpus.loc[corpus_id]
  answer['tthc'] = info['Question']

  request = chat_prompt.format_prompt(user_question=query,procedure_name= answer['tthc'], procedure = info['Answer']).to_messages()

  try:
    result = chat(request).content
  except:
    result = chat_plus(request).content

  answer['answer'] = result
  answer['reference'] = info['Link']


  related = retrieval(answer['tthc'] , question_corpus, seg_question_corpus, models)
  answer['related'] = []
  if len(retri_result) > 1:
    for i in range(1, min(6, len(retri_result))):
      corpus_id = retri_result[i]['corpus_id']
      #corpus_id = retri_result[i][1]
      link = source_corpus.loc[corpus_id, 'Link']
      question = source_corpus.loc[corpus_id, 'Question']
      answer['related'].append((question, link))
  return answer


# Load db đã có
client = chromadb.PersistentClient(path="./source")
collection_input = client.get_collection(name="Cache_chatbot_input")
collection_output = client.get_collection(name="Cache_chatbot_output")
def Find_cache(query):
  results = collection_input.query(query_texts= query, n_results=1)
  if len(results['distances'][0]) > 0 and results['distances'][0][0] < 0.1:
    id = results['ids'][0][0]
    answer = collection_output.get(ids = [id])['documents'][0]
    return 1, answer
  
  return 0, None

def Add_cache(query, answer):
  collection_input.upsert(documents=[query], ids= [str(collection_input.count())])
  collection_output.upsert(documents=[answer], ids= [str(collection_output.count())])

def Chatbot(query):
  # thêm dấu cấu
  query = corrector(query)[0]['generated_text']
  
  if Cls_in_out(query, model_cls_in_out, vectorizer_in_out) == 1:
    cache, answer = Find_cache(query)
    if cache == 0:
      a = Langchain_RAG(query)
      related =  a['related']
      src_link = f"<a href='{a['reference']}' target='_blank'>{a['tthc']}</a>"
      answer =  f"{a['answer']}\nNguồn: {src_link}"
      if len(related) > 0:
          answer += '\nCác câu hỏi và thủ tục hành chính liên quan'
          # Chuyển đổi các đường dẫn thành thẻ <a>
          for i, (label, url) in enumerate(related, start=1):
              link_html = f'<a href="{url}" target="_blank">{label}</a>'
              answer = answer + "\n" + str(i) + ". " + link_html
          answer = answer.replace("\n", '<br>')
      #answer =  f"Câu hỏi: {query}\nTên của thủ tục hành chính: {a['tthc']}\nCâu trả lời:\n{a['answer']}\nNguồn: {a['reference']}\nCác câu hỏi và thủ tục hành chính liên quan:\n{str_related}"
      
      Add_cache(query, answer)
  else:
    answer = f"Tôi là chatbot hỗ trợ giải đáp dịch vụ công, rất mong có thể giúp bạn giải đáp thắc mắc có liên quan."

  return answer
#print(Cls_in_out("quy trình xin giấy chứng nhận HIV", model_cls_in_out, vectorizer_in_out))
print(Chatbot("quy trình xin giấy chứng nhận HIV"))
