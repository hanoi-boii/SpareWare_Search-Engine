# SpareWare_Search-Engine
Course Project (Engineering Design & Development)  
Semester V, December 2020  
Skills: Natural Language Processing, Deep Learning, Web Scraping, Python  

SpareWare'20 was an e-commerce concept for electronic circuits and other hardware, involving web development, database management for user details, a search engine, recommendation system, and a payment mechanism. My responsibilities included dataset preparation and implementation of the search engine.  
  
For readying the dataset, I scraped exhaustive product data from "evelta.com" and "robu.in", and then segregated the data into 8 main categories namely "Integrated-Circuits(ICs)", "Boards-Kits", "Passive-Components", "Communication", "Sensors", "Connectors", "Optoelectronics" and "Electromechanical" components. Each of these categories was further subdivided into more specific groups to narrow down the search. For example:  
1) Integrated-Circuits --> Microcontrollers, FPGA etc.
2) Passive-Components --> Capacitor, Resistor, Inductor etc.
3) Communication --> Bluetooth, USB, GPS etc. and so on for the remaining 5 categories.

Once dataset was ready, we used one-hot encoding and ELMo for feature extraction, and a Bi-directional LSTM model followed by 3 activation hidden layers for training the data. There were a total of 4,236,652 trainable parameters.  
For any query search performed by the customer, we processed the user-input by removing punctuations and tokenizing the query. The Deep Learning model then helps identify the main category, followed by another nested model that pinpoints the sub-class. This result is then used to locate all products within the sub-class in the database, and the retrieved results are then displayed along with product name, manufacturer and cost.  
  
Demonstration link:  
https://drive.google.com/file/d/1ulBqgUjdVU5FDOivYddrawOPJSfBTgM6/view?usp=sharing  
  
For more project details, please refer "SpareWare.pdf"  
Datasets and pickled Deep Learning models available at below path:  
https://drive.google.com/drive/folders/13oQQT2CbmoXXLX9uu0cbvwG2wTCBY4MS?usp=sharing  
https://drive.google.com/drive/folders/1OMuCGPokC2uv0wJgzeKlv-O84DrqAvN-?usp=sharing  
https://drive.google.com/drive/folders/1sS1evwT6lOWD3ZFbOc6rr_esfBQIqIDC?usp=sharing
