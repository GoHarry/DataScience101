# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def convert_to_str(arg):
    '''Returns argument in string type.'''
    return str(arg)

def check_dupes(list_1,list_2):
    '''Returns duplicate values after comparing both lists.'''
    return [item for item in list_1 if item in list_2]

def pancakes(list_3):
    '''Returns list with first value as pancakes.'''
    list_3[0]="pancakes"
    return list_3
