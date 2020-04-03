# a = "abcabc"
# i = 3
# print(a[:i], a[i+1:])
#

def clean_text(text):
    text = text.replace("\n", " ").replace("BEG__", "").replace("__END", "")
    return text


text = "Summary\n\nA male newborn was apparently well until his second day of life , when BEG__increased irritability__END and a BEG__swelling in his right leg__END were noted .\nHe was rooming - in with his mother since birth .\nOn BEG__examination__END , a BEG__mass__END on the anterior surface of the right leg was noticed .\nThe BEG__mass__END was firm , elongated , ill - defined , unmovable and BEG__painful__END at BEG__palpation__END .\nNo BEG__overlying skin changes__END were seen .\nThe newborn had a family history of BEG__neonatal bone swelling__END with resolution before the age of 2 .\nBEG__Subsequent images__END showed BEG__hyperostosis in the diaphysis of the right tibia__END .\nAfter exclusion of other conditions such as BEG__trauma__END , BEG__osteomyelitis__END and BEG__congenital syphilis__END , the BEG__involvement of the tibial diaphysis__END , sparing the epiphyses and the BEG__benign course of the disease__END in family history , were indicative of BEG__Caffey disease__END .\nThe BEG__genetic study__END confirmed this diagnosis .\nBEG__Caffey disease__END , although rare , should not be overlooked in the diagnostic approach to BEG__childhood bone swelling__END .\n\nBackground\n\nBEG__Caffey disease__END is a rare cause of BEG__bone swelling__END in a newborn .\nThe manifestations of this BEG__disorder__END can sometimes resemble those of BEG__child 's physical abuse__END , as well as BEG__other diseases__END .\nFor this reason , a correct diagnosis of the cause of BEG__unexplained trauma__END , BEG__fractures__END or BEG__swellings__END in children requires that clinicians be familiar with BEG__such diseases__END , showing the importance of well - interpreted BEG__imaging studies__END in these diagnoses .\n1 , 2\n\nCase presentation\n\nA male newborn was apparently well until 28 h of life when his mother identified a BEG__swelling in his right leg__END ( figure 1 ) .\nHe was BEG__irritable__END since then .\nApart from the BEG__painful swollen leg__END , the newborn had no other BEG__systemic symptoms__END such as BEG__fever__END or BEG__poor feeding__END .\n\nBEG__Tumefaction__END in the middle third of the anterior surface of the right leg .\n\nOn BEG__examination__END , there was a BEG__firm tumefaction__END in the middle third of the anterior surface of the right leg , BEG__painful__END on BEG__palpation__END , with no BEG__changes in colour or temperature of skin over the affected limb__END .\nApart from this , the BEG__physical examination__END was normal .\nThere were no BEG__external wounds__END or BEG__bruises__END .\n\nThe mother was 33 - year - old , ARh + and gravida 3 para 2 .\nThe current pregnancy was planned and uneventful .\nBEG__Biochemical screening__END was negative .\nThree BEG__routine fetal ultrasounds__END were normal .\nIn the third trimester ( 34 weeks of gestational age ) , serological maternal investigation was irrelevant ("


print(clean_text(text))