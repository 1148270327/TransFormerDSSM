
# 引入外部库
import json

# 引入内部库
from Domain.TransformerDSSM import *


# 全局变量

def dssm_model_train (faq_dict, embedding_dict):
	"""
	dssm模型训练函数，从指定路径加载数据
	:param faq_dict:
	:param embedding_dict:
	:return:
	"""
	# 训练数据获取
	query_set = []
	answer_set = []
	for data in faq_dict:
		query_set.append(list(data['问题']))
		answer_set.append(list(data['答案']))

	# 翻转问题，增加数据多样性，变成两个问题指向同一答案
	# for key in faq_dict:
	# 	query_set.append(list(faq_dict[key]['问题'])[::-1])
	# 	answer_set.append(list(faq_dict[key]['答案'])[::-1])

	# 字向量字典获取
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1

	# 模型训练
	dssm = TransformerDSSM(q_set=query_set, t_set=answer_set, dict_set=word_dict, vec_set=vec_set, batch_size=250, is_sample=True)
	dssm.init_model_parameters()
	dssm.generate_data_set()
	# dssm.build_graph_by_gpu(4)
	dssm.build_graph_by_cpu()
	dssm.train()


def dssm_model_infer (queries, answer_embedding, embedding_dict, top_k=1, threshold=0.):
	"""
	dssm模型计算函数，通过参数获取问题，从指定路径加载需要匹配数据, 获取top-k个候选答案
	并根据给定阈值过滤答案
	:param queries:
	:param answer_embedding:
	:param top_k:
	:param threshold:
	:return:
	"""
	# 问题格式转换
	query_set = []
	for query in queries:
		query_set.append(list(query))

	# 字向量字典获取
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1

	# 模型计算
	dssm = TransformerDSSM(q_set=query_set, t_set=answer_embedding, dict_set=word_dict, vec_set=vec_set, is_train=False)
	dssm.init_model_parameters()
	dssm.generate_data_set()
	dssm.build_graph_by_cpu()
	dssm.start_session()
	result_prob_list, result_id_list = dssm.inference(top_k)
	answer_id_list = []
	for i in range(len(result_id_list)):
		answer_id = []
		for j in range(len(result_id_list[i])):
			if result_prob_list[i][j] <= threshold:
				break

			answer_id.append((result_id_list[i][j], result_prob_list[i][j]))
		answer_id_list.append(answer_id)

	return answer_id_list


def dssm_model_extract_t_pre(faq_dict, embedding_dict):
	"""
	表示
	:param embedding_dict:字向量
	:return:
	"""
	# 匹配数据获取
	t_set = []
	for key in faq_dict:
		# t_set.append(list(faq_dict[key]['答案']))
		t_set.append(list(key))


	# 字向量字典获取
	word_dict = {}
	vec_set = []
	i = 0
	for key in embedding_dict:
		word_dict[key] = i
		vec_set.append(embedding_dict[key][0])
		i += 1

	# 模型计算
	dssm = TransformerDSSM(t_set=t_set, dict_set=word_dict, vec_set=vec_set, is_extract=True)
	dssm.init_model_parameters()
	dssm.generate_data_set()
	dssm.build_graph_by_cpu()
	t_state = dssm.extract_t_pre()
	return t_state


if __name__ == '__main__':
	'''
	# ------------------- infer ----------------------------
	queries = ['公司名下没有车，车的修理费发票能不能用']
	candidates = ['公司名下没有车，取得汽车修理厂的发票可以入公帐吗？', '像汽车4S店缴营业税 税率是多少呢？', '烩面起源于河南！', '在公司名下没车，那车的维修费发票可以使用吗', '中国武汉', '最早的温度计是在1603年由意大利科学家伽利略发明的', '我们湖北哪里的面食最出名？是什么？']

	with open('./Domain/CharactersEmbedding.json', 'r', encoding='utf-8') as file_object:
		embedding_dict = json.load(file_object)

	answer_embedding = dssm_model_extract_t_pre(candidates, embedding_dict)

	xx = dssm_model_infer(queries=queries, answer_embedding=answer_embedding, embedding_dict=embedding_dict, top_k=3, threshold=0.1)
	i = 0
	for per in xx:
		print('-------------------------------')
		print('问题：'+queries[i] + '\n')
		i = i+1
		for k, v in per:
			print(str(v)+': '+candidates[k])


	'''
    # ------------- train-----------------------

	with open('./TrainData/SimFAQ.json', 'r', encoding='utf-8') as file_object:
		faq_dict = json.load(file_object)

	with open('./Domain/CharactersEmbedding.json', 'r', encoding='utf-8') as file_object:
		embedding_dict = json.load(file_object)

	dssm_model_train(faq_dict, embedding_dict)

