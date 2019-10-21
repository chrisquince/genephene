# genepheneMS

Trait prediction given annotated KOs:
```
python3 ./genephene_genome_predict.py -i test_user_data/ko_test.csv -o ko_pred.csv 
```

or Pfams:
```
python3 ./genephene_genome_predict.py -i test_user_data/pfam_test.csv -o pfam_pred.csv -g Pfam
```

This uses the sparse logistic regression classifier. For other models use the IPython Notebooks.
Input format is comma seperated and first column must be named 'genome_ID' then the KO or Pfam identifiers.

# Ipython notebooks

To run these a data file will have to be uncompressed first:

```
cd datafiles
gunzip taxid_ko_matrix_all_full.csv.gz
cd ..
```  
