# Please change all file paths with respect to direcory locations on your system



# Loading in the retrieval datasets
pickled_file = '\query_details.pickle'
pickled_file_1 = '\shuffled_product_details.pickle'
pickled_file_2 = '\list_data_final0.pickle'
with open(pickled_file, 'rb') as f:
    encodings, one_hot, word_id_y_main, puncts = pickle.load(f)
with open(pickled_file_1, 'rb') as f:
    shuffled_product_details, query_classes, query_set, query_info, query_info_list, tokenized_data, tokenized_clean_data, padded_clean_data, query_family = pickle.load(f)
with open(pickled_file_2, 'rb') as f:
    list_data_final0 = pickle.load(f)
    

# Preprocessing Helper Functions
def remove_puncts(data, puncts):
    if type(data) == list:
        for i in data:
            for j in i:
                if j in puncts:
                    i = i.replace(j, "")
    elif type(data) == str:
        for i in data:
            if i in puncts:
                data = data.replace(i, "")
    return data

def query_processing(query, puncts):
    token = word_tokenize(query)
    query = remove_puncts(token, puncts)
    query = ' '.join(query)
    return [query]
    
def sentence_to_indices(query, word_id_x, max_query_len):
    m = query.shape[0]
    query_indices = np.zeros((m, max_query_len), dtype = np.float64)
    for i in range(m):
        sentence_words = [word for word in query[i].split()]
        j = 0
        for w in sentence_words:
            if w in word_id_x:
                query_indices[i,j] = word_id_x[w]
                j+=1
    return query_indices
    
    
# Loading in some more retrieval datasets
pickled_file = '\query_details_main.pickle'   
with open(pickled_file, 'rb') as f:
    tokenized_alpha_clean_data, tokenized_alnum_clean_data, untokenized_data, word_embeddings, max_query_len, word_id_x, id_word_x, one_hot = pickle.load(f)
    
pickle_file_y_id = "\y_id_details.pickle"
with open(pickle_file_y_id, 'rb') as f:
    tokenized_alpha_clean_data, tokenized_alnum_clean_data, untokenized_data, word_embeddings, max_query_len, word_id_x, id_word_x, one_hot = pickle.load(f)
    
    
# Defining the inner model (i.e sub-classification model)
def inner_model():
    if res == 'Integrated-Circuits(ICs)':
        sub_query_final = sentence_to_indices(query, word_id_x_ic, max_query_len_ic)
        sub_prediction = model_ic.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_ic:
            if word_id_y_ic[i] == sub_max_arg:
                sub_res = i
                
    elif res == 'Boards-Kits':
        sub_query_final = sentence_to_indices(query, word_id_x_boards, max_query_len_boards)
        sub_prediction = model_boards.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_boards:
            if word_id_y_boards[i] == sub_max_arg:
                sub_res = i
                
    elif res == 'Passive-Components':
        sub_query_final = sentence_to_indices(query, word_id_x_passive, max_query_len_passive)
        sub_prediction = model_passive.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_passive:
            if word_id_y_passive[i] == sub_max_arg:
                sub_res = i
                
    elif res == 'Communication':
        sub_query_final = sentence_to_indices(query, word_id_x_comm, max_query_len_comm)
        sub_prediction = model_comm.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_comm:
            if word_id_y_comm[i] == sub_max_arg:
                sub_res = i
                
    elif res == 'Sensors':
        sub_query_final = sentence_to_indices(query, word_id_x_sensors, max_query_len_sensors)
        sub_prediction = model_sensors.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_sensors:
            if word_id_y_sensors[i] == sub_max_arg:
                sub_res = i
                
    elif res == 'Connectors':
        sub_query_final = sentence_to_indices(query, word_id_x_connectors, max_query_len_connectors)
        sub_prediction = model_comm.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_connectors:
            if word_id_y_connectors[i] == sub_max_arg:
                sub_res = i
                
    elif res == 'Optoelectronics':
        sub_query_final = sentence_to_indices(query, word_id_x_opto, max_query_len_opto)
        sub_prediction = model_opto.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_opto:
            if word_id_y_opto[i] == sub_max_arg:
                sub_res = i
                
    elif res == 'Electromechanical':
        sub_query_final = sentence_to_indices(query, word_id_x_electro, max_query_len_electro)
        sub_prediction = model_electro.predict(sub_query_final)
        sub_max_arg = np.argmax(sub_prediction)
        for i in word_id_y_electro:
            if word_id_y_electro[i] == sub_max_arg:
                sub_res = i
    return sub_res

def retrieve_ans():
    hits = []
    for i in list_data_final0:
        if (i['Class'] == sub_res) or (i['Family'] == sub_res) or (i['Sub-family'] == sub_res):
            hits.append(i)
    return hits


# Loading in the product_details
pickled_file = '\query_details_main.pickle'    
with open(pickled_file, 'rb') as f:
    tokenized_alpha_clean_data, tokenized_alnum_clean_data, untokenized_data, word_embeddings, max_query_len, word_id_x, id_word_x, one_hot = pickle.load(f)

pickled_file_ic = '\query_details_ic.pickle'
with open(pickled_file_ic, 'rb') as f:
    tokenized_alpha_clean_data_ic, tokenized_alnum_clean_data_ic, untokenized_data_ic, word_embeddings_ic, max_query_len_ic, word_id_x_ic, id_word_x_ic, one_hot_ic = pickle.load(f)

pickled_file_boards = '\query_details_boards.pickle'
with open(pickled_file_boards, 'rb') as f:
    tokenized_alpha_clean_data_boards, tokenized_alnum_clean_data_boards, untokenized_data_boards, word_embeddings_boards, max_query_len_boards, word_id_x_boards, id_word_x_boards, one_hot_boards = pickle.load(f)

pickled_file_passive = '\query_details_passive.pickle'
with open(pickled_file_passive, 'rb') as f:
    tokenized_alpha_clean_data_passive, tokenized_alnum_clean_data_passive, untokenized_data_passive, word_embeddings_passive, max_query_len_passive, word_id_x_passive, id_word_x_passive, one_hot_passive = pickle.load(f)

pickled_file_comm = '\query_details_comm.pickle'
with open(pickled_file_comm, 'rb') as f:
    tokenized_alpha_clean_data_comm, tokenized_alnum_clean_data_comm, untokenized_data_comm, word_embeddings_comm, max_query_len_comm, word_id_x_comm, id_word_x_comm, one_hot_comm = pickle.load(f)

pickled_file_sensors = '\query_details_sensors.pickle'
with open(pickled_file_sensors, 'rb') as f:
    tokenized_alpha_clean_data_sensors, tokenized_alnum_clean_data_sensors, untokenized_data_sensors, word_embeddings_sensors, max_query_len_sensors, word_id_x_sensors, id_word_x_sensors, one_hot_sensors = pickle.load(f)

pickled_file_connectors = '\query_details_connectors.pickle'
with open(pickled_file_connectors, 'rb') as f:
    tokenized_alpha_clean_data_connectors, tokenized_alnum_clean_data_connectors, untokenized_data_connectors, word_embeddings_connectors, max_query_len_connectors, word_id_x_connectors, id_word_x_connectors, one_hot_connectors = pickle.load(f)
    
pickled_file_opto = '\query_details_opto.pickle'
with open(pickled_file_opto, 'rb') as f:
    tokenized_alpha_clean_data_opto, tokenized_alnum_clean_data_opto, untokenized_data_opto, word_embeddings_opto, max_query_len_opto, word_id_x_opto, id_word_x_opto, one_hot_opto = pickle.load(f)    
    
pickled_file_electro = '\query_details_electro.pickle'
with open(pickled_file_electro, 'rb') as f:
    tokenized_alpha_clean_data_electro, tokenized_alnum_clean_data_electro, untokenized_data_electro, word_embeddings_electro, max_query_len_electro, word_id_x_electro, id_word_x_electro, one_hot_electro = pickle.load(f)        
    
    
# Loading in the models
pickle_file_main = "\wv_main.pkl"
pickle_file_ic = "\wv_ic.pkl"
pickle_file_boards = "\wv_boards.pkl"
pickle_file_passive = "\wv_passive.pkl"
pickle_file_comm = "\wv_comm.pkl"
pickle_file_sensors = "\wv_sensors.pkl"
pickle_file_connectors = "\wv_connectors.pkl"
pickle_file_opto = "\wv_opto.pkl"
pickle_file_electro = "\wv_electro.pkl"

model = joblib.load(pickle_file_main)
model_ic = joblib.load(pickle_file_ic)
model_boards = joblib.load(pickle_file_boards)
model_passive = joblib.load(pickle_file_passive)
model_comm = joblib.load(pickle_file_comm)
model_sensors = joblib.load(pickle_file_sensors)
model_connectors = joblib.load(pickle_file_connectors)
model_opto = joblib.load(pickle_file_opto)
model_electro = joblib.load(pickle_file_electro)


# Query Parsing Function
def query_parse(q):
    query_input = []

    predefined_query_fields = [{'class': 'Integrated-Circuits(ICs)','sub_class': 'Micrcontrollers', 'queries': ['integrated-circuits', 'ic', 'ics', 'integrated circuits']},
                          {'class': 'Boards-Kits','sub_class': 'Single-Board-Computers', 'queries': ['raspberry-pi', 'raspberry pi', 'elctronics kits', 'electronic kits', 'electronic boards']},
                          {'class': 'Passive-Components','sub_class': 'Passive-Components', 'queries': ['passive components', 'transistors', 'diodes', 'transistor', 'diode']},
                          {'class': 'Optoelectronics', 'sub_class': 'LEDs', 'queries': ['led', 'leds']}]
    predefined_queries = ['integrated-circuits', 'ic', 'ics', 'integrated circuits','raspberry-pi', 'raspberry pi', 'elctronics kits', 'electronic kits', 'electronic boards',
                     'passive components', 'transistors', 'diodes', 'transistor', 'diode', 'led', 'leds']
        
    if q.lower() in predefined_queries:
        for i in predefined_query_fields:
            if q.lower() in i['queries']:
                res = i['class']
                sub_res = i['sub_class']
        hits = retrieve_ans()
    else:
        query_input.append(q)
        query = query_input[0].lower()
        query = query_processing(query, puncts)
        query = np.array(query)
        query_final = sentence_to_indices(query, word_id_x, max_query_len)
        prediction = model.predict(query_final)
        max_arg = np.argmax(prediction)
        for i in word_id_y_main:
            if word_id_y_main[i] == max_arg:
                res = i
        sub_res = inner_model()
        hits = retrieve_ans()

    tokenized_hits = []
    tokenized_query = word_tokenize(q.lower())

    for i in hits:
        i['tokenized_info'] = word_tokenize(i['Info'].lower())

    for i in hits:
        match_cntr = len(set(tokenized_query).intersection(i['tokenized_info']))
        if len(tokenized_query)>1:
            if match_cntr>=(len(tokenized_query)-1):
                tokenized_hits.append(i)
        else:
            tokenized_hits.append(i)
    
    return tokenized_hits