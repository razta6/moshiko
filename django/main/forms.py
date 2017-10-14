from django import forms
from choices import *


class QueryForm(forms.Form):
	#image_title = forms.CharField(label='Enter your image query:', max_length=100)
	image_query = forms.FileField(label="Upload query image")
	num_of_results = forms.ChoiceField(label="Number of results", choices=RESULTS_CHOICES, initial='', widget=forms.Select(), required=True )
	hm_lvl = forms.ChoiceField(label="Heat Map filter level", choices=HM_CHOICES, initial='', widget=forms.Select(), required=True )
	certainty = forms.ChoiceField(label="Certainty level", choices=CERTAINTY_CHOICES, initial='', widget=forms.Select(), required=True )