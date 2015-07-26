----------------------------------------------------------------
Self-Taught Hashing for Fast Similarity Search
----------------------------------------------------------------
Dell Zhang

Matlab
%
clc; clear;
%
dataset = 'Reuters21578';
dataset = '20Newsgroups';
dataset = 'TDT2';
%
prepare_dataset(dataset,25);
%
doExperiment(dataset,'STH');
show_plot(dataset,'STH');
%
doExperiment(dataset,'STHs');
show_plot(dataset,'STHs');
%
doExperiment(dataset,'LSI');
%
doExperiment(dataset,'LCH');
%
doExperiment(dataset,'SpH');
%
show_results(dataset);
%
