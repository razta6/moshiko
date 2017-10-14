# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage #Import for storing uploaded file

from .forms import QueryForm
from django.conf import settings # import MEDIA_ROOT, MEDIA URL 

#DEVELOPMENT
from final import run

import numpy as np
from PIL import Image, ImageFilter
import os

# Create your views here.


#render the index.html page
def index(request):

	context = {}



	if request.method == 'POST':
		form = QueryForm(request.POST, request.FILES)
		if form.is_valid():
			return test(request)
	else:
		form = QueryForm()
	
	context['form'] = form

	return render(request, 'main/index.html', context)


#render the test.html page
def test(request):

	context = handle_request(request)

	return render(request, 'main/test.html', context)



###
def handle_request(request):
	
	#handled: a dict with all the parameters to be used in rendering the result page
	handled = {}
	handled['result_image_path'] = None

	fields = ['image_title', 'num_of_results', 'hm_lvl']

	for field in fields:
		handled[field] = handle_query(request, field)

	#process the inmage query - save the image in the server and extract its url and file name
	handled['image_query'], filename = handle_uploaded_file(request.FILES['image_query'])


	#Benny's stuff
	
	#HERE GOES THE MAGIC CODE TO CREATE THE RESULT
	im_path = os.path.join(settings.MEDIA_ROOT, filename)

	#run the GRAD-CAM algorithm
	results = run(settings.MEDIA_ROOT, filename, int(handled['num_of_results']), int(handled['hm_lvl']))


	if not results.err_code==0:
		#TODO: do something here
		#inside it prints to stderr...
		print("Error: {}".format(results.err_code))


	handled['err_code'] = results.err_code
	handled['err_msg'] = results.err_msg

	handled['result_top_id'] = results.top1_id.replace("_"," ")
	handled['result_other_ids'] = [name.replace("_"," ") for name in results.top_other_ids]
	handled['result_image_path'] = settings.MEDIA_URL + results.output_filename
	handled['result_features'] = list(results.s_features)

	print handled

	#DEVELOPMENT
	#handled['result_image_path'] = handled['image_query']

	return handled

### arg: a string of the request name field to be handled
def handle_query(request, field):
	query = None
	if request.POST.get(field):
		query = request.POST[field]
	return query

###
def handle_uploaded_file(myfile):
	fs = FileSystemStorage()
	filename = fs.save(myfile.name, myfile) #myfile.name
	uploaded_file_url = fs.url(filename)
	return uploaded_file_url, filename

