"""Data/decoding contracts; no model download or neural inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[1]/'src/ner_finetune.py'
spec=importlib.util.spec_from_file_location('recipe',MODULE);recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
SCHEMA={'names':{'PRICE':'private price','TERMS':'private terms'},'descriptions':{'private price':'An individually agreed nonpublic price.','private terms':'Individual payment terms.'}}

class DataTests(unittest.TestCase):
    def test_repeated_value_uses_only_explicit_occurrence(self):
        text='Открытая 10 EUR. Закрытая 10 EUR.';start=text.rindex('10 EUR')
        row={'id':'one','text':text,'spans':[{'label':'PRICE','start':start,'end':start+6,'text':'10 EUR'}],'family':'not-model-input','contrast_role':'private'}
        out=recipe.to_example(row,SCHEMA)
        self.assertEqual(out['output']['entities']['private price'],[{'text':'10 EUR','start':start,'end':start+6}])
        self.assertEqual(out['output']['entities']['private terms'],[])
        self.assertEqual(set(out),{'input','output'})
        self.assertNotIn('family',json.dumps(out))
    def test_negative_still_contains_every_schema_type(self):
        out=recipe.to_example({'text':'Публичная цена 10 EUR.','spans':[]},SCHEMA)
        self.assertEqual(list(out['output']['entities']),['private price','private terms'])
        self.assertTrue(all(not values for values in out['output']['entities'].values()))
    def test_missing_annotations_and_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'train.jsonl'
            p.write_text(json.dumps({'id':'x','text':'Данные'})+'\n')
            with self.assertRaisesRegex(ValueError,'Missing spans'):recipe.read_rows(p,SCHEMA)
            row={'id':'x','text':'Данные','spans':[]};p.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'duplicate'):recipe.read_rows(p,SCHEMA)
    def test_unknown_label_wrong_offset_overlap_fail(self):
        valid={'label':'PRICE','start':0,'end':3,'text':'10%'}
        for change in [{'label':'OTHER'},{'end':2},{'start':True}]:
            with self.assertRaises(ValueError):recipe.validate_spans('10%',[{**valid,**change}],SCHEMA['names'])
        with self.assertRaisesRegex(ValueError,'overlapping'):recipe.validate_spans('10%', [valid,{'label':'TERMS','start':0,'end':2,'text':'10'}],SCHEMA['names'])
    def test_schema_must_have_unique_names_and_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'schema.json';p.write_text(json.dumps({'names':{'P':'price','Q':'price'},'descriptions':{'price':'value'}}))
            with self.assertRaises(ValueError):recipe.read_schema(p)
    def test_greedy_overlap_and_right_to_left_mask(self):
        text='Цена 10 EUR, срок 30 дней.'
        a=text.index('10 EUR');b=text.index('30 дней')
        candidates=[{'label':'PRICE','text':'10 EUR','start':a,'end':a+6,'score':.9}, {'label':'TERMS','text':'30 дней','start':b,'end':b+7,'score':.8}, {'label':'PRICE','text':'10','start':a,'end':a+2,'score':.7}]
        pred=recipe.decode(candidates,.8)
        self.assertEqual(len(pred),2);self.assertEqual(recipe.mask(text,pred),'Цена [PRICE], срок [TERMS].')
        self.assertEqual(len(recipe.decode(candidates,.85)),1)
        with self.assertRaises(ValueError):recipe.decode(candidates,float('nan'))
    def test_exact_metric_requires_label_and_both_boundaries(self):
        gold={'label':'PRICE','start':0,'end':3,'text':'10%'};rows=[{'spans':[gold]}]
        self.assertEqual(recipe.exact_counts(rows,[[gold]])['f1'],1)
        self.assertEqual(recipe.exact_counts(rows,[[{**gold,'label':'TERMS'}]])['f1'],0)
        with self.assertRaises(ValueError):recipe.exact_counts(rows,[])
    def test_checkpoint_integrity_rejects_foreign_or_changed_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'adapter_model.safetensors').write_bytes(b'fixture')
            meta={'mode':'lora','base_id':recipe.BASE_ID,'base_revision':recipe.BASE_REVISION,'weights_sha256':{'adapter_model.safetensors':recipe.sha(p/'adapter_model.safetensors')}}
            recipe.save(p/'recipe.json',meta);self.assertEqual(recipe.read_checkpoint(p)['mode'],'lora')
            (p/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):recipe.read_checkpoint(p)

if __name__=='__main__':unittest.main()
