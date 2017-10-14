# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models
from choices import *

class Query(models.Model):
	image_query = models.FileField()
	num_of_results = models.IntegerField(choices=RESULTS_CHOICES, default=1)
	hm_lvl = models.IntegerField(choices=HM_CHOICES, default=0)
	certainty = models.IntegerField(choices=CERTAINTY_CHOICES, default=5)




