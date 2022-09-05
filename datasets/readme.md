# datasets 
## used to predict arguments containing narratives and experiences

The three complete datasets can be found in `whole_datasets.zip`.

Every dataset contains a column *label* which represents the binary classification into whether an argumentative text contains a story / personal experience (1) or not (0).
The labels from the original datasets were called __testimony__ (CMV, RegRoom) and __storytelling__ (Europolis).

The corresponding text (either online comment or transcription of spoken contribution in discussion) is stored in the column *post_text*.

The rest of the columns correspond to automatically extracted linguistic features. Details about the extracted features can be found in Tables 5-10 in the Appendix. Each column name corresponds to the feature name (first column of each Table in the Appendix).

### Change my view (CMV)

The file `cmv_all_features.csv` contains 344 comments from the dataset from the Change my View Reddit. The comments contains arguments which have been labelled with __testimony__ if they contain a personal experience (corresponds to 1 in the column label). On CMV, users exchange views on a variety of different topics and can reward other comments if they are convincing or have led to a change of their opinion. 

The original dataset was released by Egawa (2019) [1].

### Regulation Room (RegRoom)

The file `regroom_all_features.csv` contains the dataset from the regulation room (http://regulationroom.org/) and is based on the Cornell eRule- making Corpus (CDCP) (Park and Cardie, 2018) [2].
It contains 725 comments from Regulation Room, discussing consumer debt collection practices in the United States.

The dataset was also annotated with __testimony__ .

### Europolis 

The file `europolis_all_features.csv` contains a total of 856 transcribed speech contributions whose original language was German, French, and Polish (only available in the English translation). We translated the German and French transcriptions into English using DeepL and used the professional English translation of the Polish data.
Europolis contains spoken contributions from a transnational poll, in which citizens from different European countries got together to discuss about the EU and the topic immigration.
The original corpus was contructed by Gerber, 2018 [3].

Europolis was annotated with __storytelling__.


## usage

<b>If you use any of the annotated datasets please do not forget to cite the corresponding references:</b>

[1] Ryo Egawa, Gaku Morio, and Katsuhide Fujita. 2019.
Annotating and analyzing semantic role of elementary units and relations in online persuasive arguments. In Proceedings of the 57th Annual Meet- ing of the Association for Computational Linguistics: Student Research Workshop, pages 422–428, Florence, Italy. Association for Computational Linguistics.

[2] Joonsuk Park and Claire Cardie. 2018. A corpus of eRulemaking user comments for measuring evaluability of arguments. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

[3] Marlène Gerber, André Bächtiger, Susumu Shikano, Simon Reber, and Samuel Rohr. 2018. Deliberative abilities and influence in a transnational deliberative poll (europolis). British Journal of Political Science, 48(4):1093–1118.

### splits

The directory `splits` contains the 10 randomly created splits for each of the three experiments. 

- *indomain* : `cmv_10splits`, `europolis_10splits`, `regroom_10splits` with training, validation and test data (*.tsv* contains the same as *.csv* but without feature columns).
- *outdomain: 2vs1*: `2vs1_cmv` contains a concatenation of Europolis and RegRoom, the model was tested on each test split of CMV. `2vs1_europolis` contains a concatenation of CMV and RegRoom and `2vs1_regroom`contains a concatenation of CMV and Europolis.
- *all*: `mixed_10splits`, training and validation data of each split from each dataset were concatenated. Contains the test data of each dataset (*test1* = Eurpolis, *test2* = RegRoom, *test3* = CMV).
