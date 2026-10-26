"""Transcribed from v1 PDF; percentages, except Table 6 values in [0,1]."""
DATASETS = ('vqa_rad', 'slake', 'pathvqa')
TABLE1 = {
 'phi3': ((70.39,85.29,79.38),(82.02,87.02,83.98),(36.88,91.80,64.43)),
 'phi35': ((69.27,90.44,82.04),(82.95,88.70,85.20),(37.42,92.89,65.24)),
}
# Closed QA only; no downstream training. Each row is a separate checkpoint.
TABLE3 = {
 'llava7b': (59.19,50.24,45.65),
 'llava7b_llavamed': (60.66,54.33,49.07),
 'phi3_base': (54.98,55.77,52.28),
 'phi3_llavamed': (70.96,60.58,53.94),
 'phi3_pubmed': (73.90,67.31,54.38),
 'phi35_base': (57.72,57.93,53.52),
 'phi35_pubmed': (74.63,69.08,54.79),
}
# ECE, temperature-scaled ECE, temperature-scaled Brier, AUROC.
TABLE6 = {
 'phi35_zero_label': ((.333,.042,.230,.636),(.117,.039,.196,.691)),
 'phi35_zero_true': ((.174,.095,.201,.727),(.294,.094,.212,.617)),
 'phi35_zero_entropy': ((.108,.037,.145,.879),(.119,.066,.148,.827)),
 'phi35_finetune_label': ((.089,.019,.077,.889),(.082,.026,.071,.895)),
 'phi35_finetune_entropy': ((.068,.024,.066,.958),(.073,.024,.063,.949)),
 'huatuo_zero_label': ((.266,.200,.231,.811),(.190,.056,.154,.768)),
 'huatuo_zero_true': ((.114,.116,.172,.828),(.082,.074,.151,.851)),
 'huatuo_zero_verbal': ((.166,.001,.166,.754),(.141,.053,.145,.808)),
 'huatuo_zero_entropy': ((.041,.044,.141,.886),(.055,.048,.126,.885)),
}
FIG3_AUC = dict(zip(DATASETS, (.83,.85,.91)))
TABLE2 = dict(zip(('image','text','sum','union'),
                 ((53.07,47.75,51.69),(53.07,54.57,46.14),
                  (53.07,54.26,47.83),(53.07,55.97,52.70))))
TABLE5 = {'recall':89.57, 'precision':98.36, 'f1':93.76}
TABLE8 = dict(zip(('ambiguity','synonyms','position','degree','disease','numbers','shapes','unseen'),
                 ((4.34,7.24),(26.08,2.89),(21.73,42.02),(4.34,5.79),
                  (17.39,14.49),(0,18.84),(0,1.44),(26.08,7.24))))
