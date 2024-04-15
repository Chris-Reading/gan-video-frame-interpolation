import importlib.util
import pathlib, os
import customtkinter as tk
import numpy as np
from customtkinter import filedialog
from tkinter import font
from PIL import Image
import random
import glob
import importlib
from dataset import VimeoDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from generate_frames import Generate
import configparser
import train_model
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
import cv2 as cv
import format_time
import matplotlib.pyplot as plt

interpolated_frames, original_frames = [], []
interpolated_frames_idx, original_frames_idx = 0, 0
generator_model_file_path, generator_class_file_path = "", ""
generated_fps = 60


          # # # # # # # # # # # # # # # # # # # #
# # # # # #    Generators Training Variables    # # # # # #
          # # # # # # # # # # # # # # # # # # # #


generator_layers = 1
generator_core_layer, generator_norm_layer, generator_activation_layer = "", "", ""
genLSTM, genLSTMBias, genLSTMBi, genLSTMDrop, genBias, genNormEps, genNormAffine, genDropout = False, False, False, 0.0, False, 0.00001, False, False
genActivationInplace, genActivationNegativeSlope, genActivationAlpha = False, 0.0, 0.0
genKernelSize, genStrideSize, genPaddingSize = "", "", ""


          # # # # # # # # # # # # # # # # # # # # # #
# # # # # #    Discriminators Training Variables    # # # # # #
          # # # # # # # # # # # # # # # # # # # # # #


discriminator_layers = 1
discriminator_core_layer, discriminator_norm_layer, discriminator_activation_layer = "", "", ""
disLSTM, disLSTMBias, disLSTMBi, disLSTMDrop, disBias, disNormEps, disNormAffine, disDropout = False, False, False, 0.0, False, 0.00001, False, False
disActivationInplace, disActivationNegativeSlope, disActivationAlpha = False, 0.0, 0.0
disKernelSize, disStrideSize, disPaddingSize = "", "", ""


          # # # # # # # # # # # # # # # # #
# # # # # #    Training Loop Variables    # # # # # #
          # # # # # # # # # # # # # # # # #

seed, deterministic_algorithms, batch_size, latent_dim, epochs, threads = int(), bool(), int(), int(), int(), int()
loss_function, optimizer, lr, beta1, beta2 = str(), str(), float(), float(), float()
save_generated_frames, batches_per_frame_save, model_save_type, model_save_count = bool(), int(), str(), int()
plot_loss, plot_preds, plot_frames = bool(), bool(), bool()
eval_mse, eval_psnr, eval_l1 = bool(), bool(), bool()

class App(tk.CTk): # App inherits from custom tkinter library
    def __init__(self): # Constructor method
        super().__init__() # Calls the parent class' constructor: tk.CTk
        print("Initialised app")

        self.title("Video Frame Interpolation") #Window title
        
        # Sets window size automatically based on screen size
        width = self.winfo_screenwidth()               
        height = self.winfo_screenheight() 
        self.geometry("%dx%d" % (width, height))
        # Sets default appearance as dark
        tk.set_appearance_mode("Dark")

        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)
        print("Loading the GUI")

        # Calls functions to add tk items to the GUI
        self.createLeftSidebar()
        self.createImageFrame()
        self.createRightSidebar()
        
        # Sets initial values on sliders, droplists etc.
        self.textConsoleLog.tag_config("textconfig_red", foreground="#EE0000")
        self.textConsoleLog.tag_config("textconfig_green", foreground="#00EE00")
        self.textConsoleLog.tag_config("textconfig_blue", foreground="cyan")
        self.textConsoleLog.tag_config("textconfig_yellow", foreground="#EEEE00")

        self.appearanceMenu.set("Dark")
        self.textConsoleLog.insert("0.0", "GUI successfully loaded.")

    def createLeftSidebar(self):
        # Makes frame to contain buttons, labels etc.
        self.leftSidebarFrame = tk.CTkFrame(self)
        self.leftSidebarFrame.grid(row=0, column=0, rowspan=4, padx=(20,0), pady=(20, 10), sticky="nsew")
        self.leftSidebarFrame.grid_rowconfigure(6, weight=1)

        # Buttons here are for using generators / displaying results
        self.buttonExampleLoad = tk.CTkButton(master=self.leftSidebarFrame, text="Interpolation Example", command=self.example_frames) # Loads example of interpolation
        self.buttonExampleLoad.grid(row=0, column=0, padx=20, pady=(20, 10)) # Sets position inside of the frame

        self.buttonGenerateModel = tk.CTkButton(master=self.leftSidebarFrame, text="Load Generator Model", command=self.load_generator) # Load generator model and class
        self.buttonGenerateModel.grid(row=1, column=0, padx=20, pady=(20, 10))

        self.buttonImageGenerate = tk.CTkButton(master=self.leftSidebarFrame, text="Generate Frames", command=self.generate_frames) # Generate interpolation from generator model and class
        self.buttonImageGenerate.grid(row=2, column=0, padx=20, pady=(20, 10))

        self.buttonFrameGIF = tk.CTkButton(master=self.leftSidebarFrame, text="Save as GIF", command=self.frame_GIF) # Generate interpolation from generator model and class
        self.buttonFrameGIF.grid(row=3, column=0, padx=20, pady=(20, 10))

        self.gifFPSLabel = tk.CTkLabel(self.leftSidebarFrame, text="FPS:")
        self.gifFPSLabel.grid(row=4, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.gifFPS = tk.CTkEntry(self.leftSidebarFrame, placeholder_text="int e.g. 60")
        self.gifFPS.grid(row=5, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.appearanceLabel = tk.CTkLabel(master=self.leftSidebarFrame, text="Appearance Mode:") # Change GUI theme
        self.appearanceLabel.grid(row=7, column=0, padx=20, pady=(10, 0))

        self.appearanceMenu = tk.CTkOptionMenu(self.leftSidebarFrame, values=["System", "Light", "Dark"], command=self.appearanceChange)
        self.appearanceMenu.grid(row=8, column=0, padx=20, pady=(10, 10))

        self.scalingLabel = tk.CTkLabel(master=self.leftSidebarFrame, text="UI Scaling:") # Change scale
        self.scalingLabel.grid(row=9, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = tk.CTkOptionMenu(self.leftSidebarFrame, values=["100%", "90%", "80%", "70%", "60%", "50%", "40%", "30%", "20%", "10%"], command=self.scalingChange)
        self.scaling_optionemenu.grid(row=10, column=0, padx=20, pady=(10, 20))

    def createImageFrame(self):
        # Middle section of GUI for displaying results and a console log
        self.labelImageLoad = tk.CTkFrame(self, corner_radius=0)
        self.labelImageLoad.grid(row=0, column=1, rowspan=4, columnspan=2, padx=(20, 0), pady=(0, 5), sticky="nsew")
        self.labelImageLoad.grid_columnconfigure(0, weight=1)
        self.labelImageLoad.grid_columnconfigure(1, weight=1)
        self.labelImageLoad.grid_rowconfigure(1, weight=1)

        self.leftimagetext = tk.CTkLabel(master=self.labelImageLoad, text="")
        self.leftimagetext.grid(row=0, column=0, pady=(50, 0))

        self.rightimagetext = tk.CTkLabel(master=self.labelImageLoad, text="")
        self.rightimagetext.grid(row=0, column=1, pady=(50, 0))

        self.buttonFrameBackward = tk.CTkButton(master=self.labelImageLoad, text="<", command=self.backwardFrames) # Button to go back a frame
        self.buttonFrameBackward.grid(row=2, column=0, padx=20, pady=(20, 10)) 
        self.buttonFrameForward = tk.CTkButton(master=self.labelImageLoad, text=">", command=self.forwardFrames) # Button to go forward a frame
        self.buttonFrameForward.grid(row=2, column=1, padx=20, pady=(20, 10))

        self.textConsoleLog = tk.CTkTextbox(master=self.labelImageLoad, width=900, corner_radius=1, text_color="white", border_color="#333", border_width=2)
        self.textConsoleLog.grid(row=3, column=0, columnspan=2)

    def createRightSidebar(self):
        # Right section for creating a new GAN model
        self.rightSidebarFrame = tk.CTkScrollableFrame(self, width=300)
        self.rightSidebarFrame.grid(row=0, column=5, padx=(20,0), pady=(20, 10), sticky="nsew")
        #self.ganTitle = tk.CTkLabel(self.rightSidebarFrame, text="Design a GAN Model:")
        #self.ganTitle.grid(row=0, column=0)

        self.buttonResetGANVars = tk.CTkButton(master=self.rightSidebarFrame, text="Restart GAN Creator", command=self.resetGAN, fg_color="red", hover_color="red") #Load initial image
        self.buttonResetGANVars.grid(row=0, column=0, padx=100, pady=(20, 10), sticky="nsew") #Sets position inside of the frame

        self.createTabview()

        self.buttonTraining = tk.CTkButton(master=self, text="Begin Training", command=self.beginTraining) #Load initial image
        self.buttonTraining.grid(row=1, column=5, padx=20, pady=(20, 10)) #Sets position inside of the frame

    def resetGAN(self):
        global generator_layers, discriminator_layers, generator_core_layer, generator_norm_layer, generator_activation_layer, discriminator_core_layer, discriminator_norm_layer, discriminator_activation_layer
        global genLSTM, genLSTMBias, genLSTMBi, genLSTMDrop, genBias, genNormEps, genNormAffine, genDropout
        global disLSTM, disLSTMBias, disLSTMBi, disLSTMDrop, disBias, disNormEps, disNormAffine, disDropout
        global genActivationInplace, genActivationNegativeSlope, genActivationAlpha
        global disActivationInplace, disActivationNegativeSlope, disActivationAlpha
        global genKernelSize, genStrideSize, genPaddingSize
        global disKernelSize, disStrideSize, disPaddingSize
        global seed, deterministic_algorithms, batch_size, latent_dim, epochs, threads
        global loss_function, optimizer, lr, beta1, beta2
        global save_generated_frames, batches_per_frame_save, model_save_type, model_save_count
        global plot_loss, plot_preds, plot_frames

        generator_layers = 1
        discriminator_layers = 1
        generator_core_layer, generator_norm_layer, generator_activation_layer, discriminator_core_layer, discriminator_norm_layer, discriminator_activation_layer = "", "", "", "", "", ""
        genLSTM, genLSTMBias, genLSTMBi, genLSTMDrop, genBias, genNormEps, genNormAffine, genDropout = False, False, False, 0.0, False, 0.00001, False, False
        genActivationInplace, genActivationNegativeSlope, genActivationAlpha = False, 0.0, 0.0
        genKernelSize, genStrideSize, genPaddingSize = "", "", ""

        disLSTM, disLSTMBias, disLSTMBi, disLSTMDrop, disBias, disNormEps, disNormAffine, disDropout = False, False, False, 0.0, False, 0.00001, False, False
        disActivationInplace, disActivationNegativeSlope, disActivationAlpha = False, 0.0, 0.0
        disKernelSize, disStrideSize, disPaddingSize = "", "", ""

        seed, deterministic_algorithms, batch_size, latent_dim, epochs, threads = int(), bool(), int(), int(), int(), int()
        loss_function, optimizer, lr, beta1, beta2 = str(), str(), float(), float(), float()
        save_generated_frames, batches_per_frame_save, model_save_type, model_save_count = bool(), int(), str(), int()
        plot_loss, plot_preds, plot_frames = bool(), bool(), bool()

        self.createTabview()
        
    def createTabview(self):
        self.tabview = tk.CTkTabview(self.rightSidebarFrame)
        self.tabview.grid(row=2, column=0, columnspan=2, rowspan=4, padx=20, pady=(5, 10), sticky="nsew")
        self.tabview.add("Generator")
        self.tabview.add("Discriminator")
        self.tabview.add("Training")
        self.tabview.tab("Generator").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Discriminator").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Training").grid_columnconfigure(0, weight=1)


                            # # # # # # # # # # # # # #
        # # # # # # # # # # #    Generator Buttons    # # # # # # # # # # #
                            # # # # # # # # # # # # # #


        self.genLayerLabel = tk.CTkLabel(self.tabview.tab("Generator"), text=f"Number of hidden layers: {generator_layers}")
        self.genLayerLabel.grid(row=0, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.genLayerSlider = tk.CTkSlider(self.tabview.tab("Generator"), from_=1, to=10, number_of_steps=9, command=self.genLayerCounter)
        self.genLayerSlider.grid(row=1, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.genCoreLabel = tk.CTkLabel(self.tabview.tab("Generator"), text="Core Layer Type:")
        self.genCoreLabel.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.genCoreMenu = tk.CTkOptionMenu(self.tabview.tab("Generator"), values=["Convolution Layer", "Dense Layer"], command=self.generatorCoreLayer)
        self.genCoreMenu.grid(row=3, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.genLSTMCheckbox = tk.CTkCheckBox(self.tabview.tab("Generator"), text="Add LSTM layer", command=self.generatorLSTMLayer)
        self.genLSTMCheckbox.grid(row=4, column=0, padx=20, pady=(5, 10))

        self.genNormLabel = tk.CTkLabel(self.tabview.tab("Generator"), text="Normalization Layer Type:")
        self.genNormLabel.grid(row=15, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.genNormMenu = tk.CTkOptionMenu(self.tabview.tab("Generator"), values=["Batch", "Instance"], command=self.generatorNormLayer)
        self.genNormMenu.grid(row=16, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.genActivationLabel = tk.CTkLabel(self.tabview.tab("Generator"), text="Activation Layer Type:")
        self.genActivationLabel.grid(row=20, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.genActivationMenu = tk.CTkOptionMenu(self.tabview.tab("Generator"), values=["ReLU", "LeakyReLU", "ELU", "Softplus"], command=self.generatorActivation)
        self.genActivationMenu.grid(row=21, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.genDropoutCheckbox = tk.CTkCheckBox(self.tabview.tab("Generator"), text="Add Dropout layer", command=self.generatorDropout)
        self.genDropoutCheckbox.grid(row=25, column=0, padx=20, pady=(5, 10))

        self.buttonTest = tk.CTkButton(master=self.tabview.tab("Generator"), text="Check parameters", command=self.genVarCheck) #Load 
        self.buttonTest.grid(row=26, column=0, padx=20, pady=(5, 10), sticky="nsew")
        
        self.genLayerSlider.set(generator_layers)


                            # # # # # # # # # # # # # # # #
        # # # # # # # # # # #    Discriminator Buttons    # # # # # # # # # # #
                            # # # # # # # # # # # # # # # #


        self.disLayerLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text=f"Number of hidden layers: {discriminator_layers}")
        self.disLayerLabel.grid(row=0, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.disLayerSlider = tk.CTkSlider(self.tabview.tab("Discriminator"), from_=1, to=10, number_of_steps=9, command=self.disLayerCounter)
        self.disLayerSlider.grid(row=1, column=0, padx=20, pady=(5, 10), sticky="nsew")
        
        self.disCoreLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text="Core Layer Type:")
        self.disCoreLabel.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.disCoreMenu = tk.CTkOptionMenu(self.tabview.tab("Discriminator"), values=["Convolution Layer", "Dense Layer", "Pooling Layer"], command=self.discriminatorCoreLayer)
        self.disCoreMenu.grid(row=3, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.disLSTMCheckbox = tk.CTkCheckBox(self.tabview.tab("Discriminator"), text="Add LSTM layer", command=self.discriminatorLSTMLayer)
        self.disLSTMCheckbox.grid(row=4, column=0, padx=20, pady=(5, 10))

        self.disNormLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text="Normalization Layer Type:")
        self.disNormLabel.grid(row=15, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.disNormMenu = tk.CTkOptionMenu(self.tabview.tab("Discriminator"), values=["Batch", "Instance"], command=self.discriminatorNormLayer)
        self.disNormMenu.grid(row=16, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.disActivationLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text="Activation Layer Type:")
        self.disActivationLabel.grid(row=20, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.disActivationMenu = tk.CTkOptionMenu(self.tabview.tab("Discriminator"), values=["ReLU", "LeakyReLU", "ELU", "Softplus"], command=self.discriminatorActivation)
        self.disActivationMenu.grid(row=21, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.disDropoutCheckbox = tk.CTkCheckBox(self.tabview.tab("Discriminator"), text="Add Dropout layer", command=self.discriminatorDropout)
        self.disDropoutCheckbox.grid(row=25, column=0, padx=20, pady=(5, 10))

        self.buttonTest = tk.CTkButton(master=self.tabview.tab("Discriminator"), text="Check parameters", command=self.disVarCheck) #Load 
        self.buttonTest.grid(row=26, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.disLayerSlider.set(discriminator_layers)


                            # # # # # # # # # # # # # # # #
        # # # # # # # # # # #    Training Loop Buttons    # # # # # # # # # # #
                            # # # # # # # # # # # # # # # #


        self.trainingSeedLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Random Seed:")
        self.trainingSeedLabel.grid(row=0, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.trainingSeed = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="int e.g. 1024")
        self.trainingSeed.grid(row=1, column=0, padx=20, pady=(5, 10), sticky="nsew") 

        self.trainingDeterministicAlgorithms = tk.CTkCheckBox(self.tabview.tab("Training"), text="Deterministic Algorithms", command=self.deterministicAlgorithms)
        self.trainingDeterministicAlgorithms.grid(row=2, column=0, padx=20, pady=(5, 10))

        self.batchSizeLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Batch Size:")
        self.batchSizeLabel.grid(row=3, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.batchSize = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="int e.g. 16")
        self.batchSize.grid(row=4, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.latentDimLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Latent Dimension:")
        self.latentDimLabel.grid(row=5, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.latentDim = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="int e.g. 100")
        self.latentDim.grid(row=6, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.numEpochsLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Number of Epochs:")
        self.numEpochsLabel.grid(row=7, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.numEpochs = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="int e.g. 5")
        self.numEpochs.grid(row=8, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.workerThreadsLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Number of Worker Threads:")
        self.workerThreadsLabel.grid(row=9, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.workerThreads = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="int e.g. 4")
        self.workerThreads.grid(row=10, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.trainingLossLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Loss Function Type:")
        self.trainingLossLabel.grid(row=11, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.trainingLossMenu = tk.CTkOptionMenu(self.tabview.tab("Training"), values=["BCE", "BCEWithLogitsLoss", "MSE"], command=self.trainingLossFunction)
        self.trainingLossMenu.grid(row=12, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.optimizerLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Optimizer Type:")
        self.optimizerLabel.grid(row=13, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.optimizerMenu = tk.CTkOptionMenu(self.tabview.tab("Training"), values=["Adam", "RMSProp", "SGD", "Adagrad", "AdamW"], command=self.trainingOptimizer)
        self.optimizerMenu.grid(row=14, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.saveGeneratedFramesCheckbox = tk.CTkCheckBox(self.tabview.tab("Training"), text="Save Generated Frames", command=self.generatedFrames)
        self.saveGeneratedFramesCheckbox.grid(row=21, column=0, padx=20, pady=(5, 10))
        self.batchesPerSaveLabel = tk.CTkLabel(self.tabview.tab("Training"), text="After how many batches:")
        self.batchesPerSaveLabel.grid(row=22, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.batchesPerSave = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="int e.g. 50")
        self.batchesPerSave.grid(row=23, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.saveModelLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Save models in:")
        self.saveModelLabel.grid(row=24, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.saveModelMenu = tk.CTkOptionMenu(self.tabview.tab("Training"), values=["Epochs", "Batches"], command=self.modelSaveType)
        self.saveModelMenu.grid(row=25, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.plotLossCheckbox = tk.CTkCheckBox(self.tabview.tab("Training"), text="Plot Model Loss", command=self.plotModelLoss)
        self.plotLossCheckbox.grid(row=28, column=0, padx=20, pady=(5, 10))
        self.plotPredsCheckbox = tk.CTkCheckBox(self.tabview.tab("Training"), text="Plot Model Predictions", command=self.plotModelPredictions)
        self.plotPredsCheckbox.grid(row=29, column=0, padx=20, pady=(5, 10))        
        self.plotFramesCheckbox = tk.CTkCheckBox(self.tabview.tab("Training"), text="Plot Generated Frames", command=self.plotGenerated)
        self.plotFramesCheckbox.grid(row=30, column=0, padx=20, pady=(5, 10))

        self.evalMSECheckbox = tk.CTkCheckBox(self.tabview.tab("Training"), text="MSE Evaluation", command=self.evalMSECheck)
        self.evalMSECheckbox.grid(row=31, column=0, padx=20, pady=(5, 10))
        self.evalPSNRCheckbox = tk.CTkCheckBox(self.tabview.tab("Training"), text="PSNR Evaluation", command=self.evalPSNRCheck)
        self.evalPSNRCheckbox.grid(row=32, column=0, padx=20, pady=(5, 10))        
        self.evalL1Checkbox = tk.CTkCheckBox(self.tabview.tab("Training"), text="L1 Evaluation", command=self.evalL1Check)
        self.evalL1Checkbox.grid(row=33, column=0, padx=20, pady=(5, 10))

        self.buttonTest = tk.CTkButton(master=self.tabview.tab("Training"), text="Check parameters", command=self.trainingVarCheck) #Load 
        self.buttonTest.grid(row=34, column=0, padx=20, pady=(5, 10), sticky="nsew")


              # # # # # # # # # # # # # #
    # # # # # #    Generator Buttons    # # # # # # # # # # # # # # # #
              # # # # # # # # # # # # # #


                        # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Core Layers Functions    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # #
    

    def generatorCoreLayer(self, new: str):
        global generator_core_layer
        generator_core_layer = new
        self.generatorCoreLayerArgs()


    def generatorCoreLayerArgs(self):
        if genLSTM:
            self.genLSTMBiasCheckbox = tk.CTkCheckBox(self.tabview.tab("Generator"), text="LSTM Additive Bias", command=self.generatorLSTMBias)
            self.genLSTMBiasCheckbox.grid(row=5, column=0, padx=20, pady=(5, 10))
            self.genLSTMBiCheckbox = tk.CTkCheckBox(self.tabview.tab("Generator"), text="BiDirectional LSTM", command=self.generatorBiDirectional)
            self.genLSTMBiCheckbox.grid(row=6, column=0, padx=20, pady=(5, 10))
            self.genLSTMDropoutLabel = tk.CTkLabel(self.tabview.tab("Generator"), text=f"Dropout probability (4d.p.): {genLSTMDrop}")
            self.genLSTMDropoutLabel.grid(row=7, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genLSTMDropoutSlider = tk.CTkSlider(self.tabview.tab("Generator"), from_=0, to=0.999, number_of_steps=999, command=self.genLSTMDropout)
            self.genLSTMDropoutSlider.grid(row=8, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genLSTMDropoutSlider.set(float(genLSTMDrop))

        if generator_core_layer == "Convolution Layer":
            self.genConvKernelLabel = tk.CTkLabel(self.tabview.tab("Generator"), text="Kernel Size:")
            self.genConvKernelLabel.grid(row=9, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genConvKernelSize = tk.CTkEntry(self.tabview.tab("Generator"), placeholder_text="E.g. 3 or (3, 2)")
            self.genConvKernelSize.grid(row=10, column=0, padx=20, pady=(5, 10), sticky="nsew")  
            
            self.genConvStrideLabel = tk.CTkLabel(self.tabview.tab("Generator"), text="Stride Size:")
            self.genConvStrideLabel.grid(row=11, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genConvStrideSize = tk.CTkEntry(self.tabview.tab("Generator"), placeholder_text="E.g. 3 or (3, 2)")
            self.genConvStrideSize.grid(row=12, column=0, padx=20, pady=(5, 10), sticky="nsew")  

            self.genConvPaddingLabel = tk.CTkLabel(self.tabview.tab("Generator"), text="Padding Size:")
            self.genConvPaddingLabel.grid(row=13, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genConvPaddingSize = tk.CTkEntry(self.tabview.tab("Generator"), placeholder_text="E.g. 3 or (3, 2)")
            self.genConvPaddingSize.grid(row=14, column=0, padx=20, pady=(5, 10), sticky="nsew")  
            
        self.genBiasCheckbox = tk.CTkCheckBox(self.tabview.tab("Generator"), text="Additive Bias", command=self.generatorBias)
        self.genBiasCheckbox.grid(row=9, column=0, padx=20, pady=(5, 10))

    def generatorLSTMBias(self):
        global genLSTMBias
        if self.genLSTMBiasCheckbox.get() == 1:
            genLSTMBias = True
        elif self.genLSTMBiasCheckbox.get() == 0:
            genLSTMBias = False

    def generatorBias(self):
        global genBias
        if self.genLSTMBiasCheckbox.get() == 1:
            genBias = True
        elif self.genLSTMBiasCheckbox.get() == 0:
            genBias = False
    
    def generatorLSTMLayer(self):
        global genLSTM
        if self.genLSTMCheckbox.get() == 1:
            genLSTM = True
        elif self.genLSTMCheckbox.get() == 0:
            genLSTM = False
        self.generatorCoreLayerArgs()

    def generatorBiDirectional(self):
        global genLSTMBi
        if self.genLSTMBiCheckbox.get() == 1:
            genLSTMBi = True
        elif self.genLSTMBiCheckbox.get() == 0:
            genLSTMBi = False        

    def genLSTMDropout(self, prob: str):
        global genLSTMDrop
        genLSTMDrop = round(float(prob), 4)
        self.genLSTMDropoutLabel.configure(text=f"Dropout probability (4d.p.): {genLSTMDrop}")


                        # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Norm Layers Functions    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # #


    def generatorNormLayer(self, new: str):
        global generator_norm_layer
        generator_norm_layer = new

        self.genNormEpsLabel = tk.CTkLabel(self.tabview.tab("Generator"), text=f"Numerical Stability Epsilon:")
        self.genNormEpsLabel.grid(row=17, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.genNormEpsSize = tk.CTkEntry(self.tabview.tab("Generator"), placeholder_text="Between 0.00001 and 0.00000001")
        self.genNormEpsSize.grid(row=18, column=0, padx=20, pady=(5, 10), sticky="nsew")  
        self.genNormAffineCheckbox = tk.CTkCheckBox(self.tabview.tab("Generator"), text="Learnable Affine Parameters", command=self.generatorNormAffine)
        self.genNormAffineCheckbox.grid(row=19, column=0, padx=20, pady=(5, 10))

    def generatorNormAffine(self):
        global genNormAffine
        if self.genNormAffineCheckbox.get() == 1:
            genNormAffine = True
        elif self.genNormAffineCheckbox.get() == 0:
            genNormAffine = False 


                        # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Activation Layers Functions    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # # # # #


    def generatorActivation(self, new: str):
        global generator_activation_layer
        generator_activation_layer = new

        if generator_activation_layer == "ReLU":
            self.genActivationInplaceCheckbox = tk.CTkCheckBox(self.tabview.tab("Generator"), text="In-place Tensors", command=self.generatorActivationInplace)
            self.genActivationInplaceCheckbox.grid(row=22, column=0, padx=20, pady=(5, 10))
        
        elif generator_activation_layer == "LeakyReLU":
            self.genActivationNegativeSlopeLabel = tk.CTkLabel(self.tabview.tab("Generator"), text=f"Negative Slope (4d.p.): {genActivationNegativeSlope}")
            self.genActivationNegativeSlopeLabel.grid(row=23, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genActivationNegativeSlopeSlider = tk.CTkSlider(self.tabview.tab("Generator"), from_=0, to=1, number_of_steps=1000, command=self.generatorActivationNegativeSlope)
            self.genActivationNegativeSlopeSlider.grid(row=24, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genActivationNegativeSlopeSlider.set(float(genActivationNegativeSlope))

        elif generator_activation_layer == "ELU":
            self.genActivationAlphaLabel = tk.CTkLabel(self.tabview.tab("Generator"), text=f"Alpha Slope (4d.p.): {genActivationAlpha}")
            self.genActivationAlphaLabel.grid(row=23, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genActivationAlphaSlider = tk.CTkSlider(self.tabview.tab("Generator"), from_=0, to=1, number_of_steps=1000, command=self.generatorActivationAlpha)
            self.genActivationAlphaSlider.grid(row=24, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.genActivationAlphaSlider.set(float(genActivationAlpha))

    def generatorActivationInplace(self):
        global genActivationInplace
        if self.genActivationInplaceCheckbox.get() == 1:
            genActivationInplace = True
        elif self.genActivationInplaceCheckbox.get() == 0:
            genActivationInplace = False 

    def generatorActivationNegativeSlope(self, val: str):
        global genActivationNegativeSlope
        genActivationNegativeSlope = round(float(val), 4)
        self.genActivationNegativeSlopeLabel.configure(text=f"Negative Slope (4d.p.): {genActivationNegativeSlope}")

    def generatorActivationAlpha(self, val: str):
        global genActivationAlpha
        genActivationAlpha = round(float(val), 4)
        self.genActivationAlphaLabel.configure(text=f"Alpha Slope (4d.p.): {genActivationAlpha}")


                        # # # # # # # # # # # #
    # # # # # # # # # # #    Dropout Layer    # # # # # # # # # # #
                        # # # # # # # # # # # #


    def generatorDropout(self):
        global genDropout
        if self.genDropoutCheckbox.get() == 1:
            genDropout = True
        elif self.genDropoutCheckbox.get() == 0:
            genDropout = False


                        # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Model Parameter Check    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # #


    def genVarCheck(self):
        global genNormEps, genNormAffine
        self.textConsoleLog.delete("0.0", "end")
        self.textConsoleLog.insert("0.0", "GAN's Generator Architecture")
        self.textConsoleLog.insert("end", "\nNumber of layers: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{generator_layers}")
        self.textConsoleLog.insert("end", "\nCore Type: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{generator_core_layer}")

        if generator_core_layer == "Convolution Layer":
            global genKernelSize, genStrideSize, genPaddingSize
            genKernelSize = self.genConvKernelSize.get()
            genStrideSize = self.genConvStrideSize.get()
            genPaddingSize = self.genConvPaddingSize.get()

            self.textConsoleLog.insert("end", "\nKernel Size: ", "textconfig_red")
            self.textConsoleLog.insert("end", f"{genKernelSize}")
            self.textConsoleLog.insert("end", "\nStride Size: ", "textconfig_red")
            self.textConsoleLog.insert("end", f"{genStrideSize}")
            self.textConsoleLog.insert("end", "\nPadding Size: ", "textconfig_red")
            self.textConsoleLog.insert("end", f"{genPaddingSize}")
        

        self.textConsoleLog.insert("end", "\nAdditive Bias: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genBias}")

        self.textConsoleLog.insert("end", "\n\nAdditional LSTM Layer: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genLSTM}")
        self.textConsoleLog.insert("end", "\nLSTM Bias: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genLSTMBias}")
        self.textConsoleLog.insert("end", "\nBiDirectional LSTM: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genLSTMBi}")
        self.textConsoleLog.insert("end", "\nLSTM Dropout probability: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genLSTMDrop}")

        genNormEps = self.genNormEpsSize.get()

        self.textConsoleLog.insert("end", "\n\nNormalization Layer Type: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{generator_norm_layer}")
        self.textConsoleLog.insert("end", "\nNumerical Stability Epsilon: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genNormEps}")        
        self.textConsoleLog.insert("end", "\nLearnable Affine Parameters: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genNormAffine}")

        self.textConsoleLog.insert("end", "\n\nActivation Layer Type: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{generator_activation_layer}")

        if generator_activation_layer == "ReLU":
            self.textConsoleLog.insert("end", "\nIn-place Tensors: ", "textconfig_red")
            self.textConsoleLog.insert("end", f"{genActivationInplace}")        
        elif generator_activation_layer == "LeakyReLU":
            self.textConsoleLog.insert("end", "\nNegative Slope: ", "textconfig_red")
            self.textConsoleLog.insert("end", f"{genActivationNegativeSlope}")    
        elif generator_activation_layer == "ELU":
            self.textConsoleLog.insert("end", "\nAlpha Slope: ", "textconfig_red")
            self.textConsoleLog.insert("end", f"{genActivationAlpha}")    

        self.textConsoleLog.insert("end", "\n\nDropout Layer: ", "textconfig_red")
        self.textConsoleLog.insert("end", f"{genDropout}")

    def genLayerCounter(self, num: str):
        global generator_layers
        generator_layers = int(num)
        self.genLayerLabel.configure(text=f"Number of hidden layers: {generator_layers}")


              # # # # # # # # # # # # # # # #
    # # # # # #    Discriminator Buttons    # # # # # # # # # # # # # # # #
              # # # # # # # # # # # # # # # #
   

                        # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Core Layers Functions    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # #
    

    def discriminatorCoreLayer(self, new: str):
        global discriminator_core_layer
        discriminator_core_layer = new
        self.discriminatorCoreLayerArgs()


    def discriminatorCoreLayerArgs(self):
        if disLSTM:
            self.disLSTMBiasCheckbox = tk.CTkCheckBox(self.tabview.tab("Discriminator"), text="LSTM Additive Bias", command=self.discriminatorLSTMBias)
            self.disLSTMBiasCheckbox.grid(row=5, column=0, padx=20, pady=(5, 10))
            self.disLSTMBiCheckbox = tk.CTkCheckBox(self.tabview.tab("Discriminator"), text="BiDirectional LSTM", command=self.discriminatorBiDirectional)
            self.disLSTMBiCheckbox.grid(row=6, column=0, padx=20, pady=(5, 10))
            self.disLSTMDropoutLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text=f"Dropout probability (4d.p.): {disLSTMDrop}")
            self.disLSTMDropoutLabel.grid(row=7, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disLSTMDropoutSlider = tk.CTkSlider(self.tabview.tab("Discriminator"), from_=0, to=0.999, number_of_steps=999, command=self.disLSTMDropout)
            self.disLSTMDropoutSlider.grid(row=8, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disLSTMDropoutSlider.set(float(disLSTMDrop))

        if discriminator_core_layer == "Convolution Layer":
            self.disConvKernelLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text="Kernel Size:")
            self.disConvKernelLabel.grid(row=9, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disConvKernelSize = tk.CTkEntry(self.tabview.tab("Discriminator"), placeholder_text="E.g. 3 or (3, 2)")
            self.disConvKernelSize.grid(row=10, column=0, padx=20, pady=(5, 10), sticky="nsew")  
            
            self.disConvStrideLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text="Stride Size:")
            self.disConvStrideLabel.grid(row=11, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disConvStrideSize = tk.CTkEntry(self.tabview.tab("Discriminator"), placeholder_text="E.g. 3 or (3, 2)")
            self.disConvStrideSize.grid(row=12, column=0, padx=20, pady=(5, 10), sticky="nsew")  

            self.disConvPaddingLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text="Padding Size:")
            self.disConvPaddingLabel.grid(row=13, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disConvPaddingSize = tk.CTkEntry(self.tabview.tab("Discriminator"), placeholder_text="E.g. 3 or (3, 2)")
            self.disConvPaddingSize.grid(row=14, column=0, padx=20, pady=(5, 10), sticky="nsew")  
            
        self.disBiasCheckbox = tk.CTkCheckBox(self.tabview.tab("Discriminator"), text="Additive Bias", command=self.discriminatorBias)
        self.disBiasCheckbox.grid(row=9, column=0, padx=20, pady=(5, 10))

    def discriminatorLSTMBias(self):
        global disLSTMBias
        if self.disLSTMBiasCheckbox.get() == 1:
            disLSTMBias = True
        elif self.disLSTMBiasCheckbox.get() == 0:
            disLSTMBias = False

    def discriminatorBias(self):
        global disBias
        if self.disLSTMBiasCheckbox.get() == 1:
            disBias = True
        elif self.disLSTMBiasCheckbox.get() == 0:
            disBias = False
    
    def discriminatorLSTMLayer(self):
        global disLSTM
        if self.disLSTMCheckbox.get() == 1:
            disLSTM = True
        elif self.disLSTMCheckbox.get() == 0:
            disLSTM = False
        self.discriminatorCoreLayerArgs()

    def discriminatorBiDirectional(self):
        global disLSTMBi
        if self.disLSTMBiCheckbox.get() == 1:
            disLSTMBi = True
        elif self.disLSTMBiCheckbox.get() == 0:
            disLSTMBi = False        

    def disLSTMDropout(self, prob: str):
        global disLSTMDrop
        disLSTMDrop = round(float(prob), 4)
        self.disLSTMDropoutLabel.configure(text=f"Dropout probability (4d.p.): {disLSTMDrop}")


                        # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Norm Layers Functions    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # #


    def discriminatorNormLayer(self, new: str):
        global discriminator_norm_layer
        discriminator_norm_layer = new

        self.disNormEpsLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text=f"Numerical Stability Epsilon:")
        self.disNormEpsLabel.grid(row=17, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.disNormEpsSize = tk.CTkEntry(self.tabview.tab("Discriminator"), placeholder_text="Between 0.00001 and 0.00000001")
        self.disNormEpsSize.grid(row=18, column=0, padx=20, pady=(5, 10), sticky="nsew")  
        self.disNormAffineCheckbox = tk.CTkCheckBox(self.tabview.tab("Discriminator"), text="Learnable Affine Parameters", command=self.discriminatorNormAffine)
        self.disNormAffineCheckbox.grid(row=19, column=0, padx=20, pady=(5, 10))

    def discriminatorNormAffine(self):
        global disNormAffine
        if self.disNormAffineCheckbox.get() == 1:
            disNormAffine = True
        elif self.disNormAffineCheckbox.get() == 0:
            disNormAffine = False 


                        # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Activation Layers Functions    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # # # # #


    def discriminatorActivation(self, new: str):
        global discriminator_activation_layer
        discriminator_activation_layer = new

        if discriminator_activation_layer == "ReLU":
            self.disActivationInplaceCheckbox = tk.CTkCheckBox(self.tabview.tab("Discriminator"), text="In-place Tensors", command=self.discriminatorActivationInplace)
            self.disActivationInplaceCheckbox.grid(row=22, column=0, padx=20, pady=(5, 10))
        
        elif discriminator_activation_layer == "LeakyReLU":
            self.disActivationNegativeSlopeLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text=f"Negative Slope (4d.p.): {disActivationNegativeSlope}")
            self.disActivationNegativeSlopeLabel.grid(row=23, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disActivationNegativeSlopeSlider = tk.CTkSlider(self.tabview.tab("Discriminator"), from_=0, to=1, number_of_steps=1000, command=self.discriminatorActivationNegativeSlope)
            self.disActivationNegativeSlopeSlider.grid(row=24, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disActivationNegativeSlopeSlider.set(float(disActivationNegativeSlope))

        elif discriminator_activation_layer == "ELU":
            self.disActivationAlphaLabel = tk.CTkLabel(self.tabview.tab("Discriminator"), text=f"Alpha Slope (4d.p.): {disActivationAlpha}")
            self.disActivationAlphaLabel.grid(row=23, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disActivationAlphaSlider = tk.CTkSlider(self.tabview.tab("Discriminator"), from_=0, to=1, number_of_steps=1000, command=self.discriminatorActivationAlpha)
            self.disActivationAlphaSlider.grid(row=24, column=0, padx=20, pady=(5, 10), sticky="nsew")
            self.disActivationAlphaSlider.set(float(disActivationAlpha))

    def discriminatorActivationInplace(self):
        global disActivationInplace
        if self.disActivationInplaceCheckbox.get() == 1:
            disActivationInplace = True
        elif self.disActivationInplaceCheckbox.get() == 0:
            disActivationInplace = False 

    def discriminatorActivationNegativeSlope(self, val: str):
        global disActivationNegativeSlope
        disActivationNegativeSlope = round(float(val), 4)
        self.disActivationNegativeSlopeLabel.configure(text=f"Negative Slope (4d.p.): {disActivationNegativeSlope}")

    def discriminatorActivationAlpha(self, val: str):
        global disActivationAlpha
        disActivationAlpha = round(float(val), 4)
        self.disActivationAlphaLabel.configure(text=f"Alpha Slope (4d.p.): {disActivationAlpha}")


                        # # # # # # # # # # # #
    # # # # # # # # # # #    Dropout Layer    # # # # # # # # # # #
                        # # # # # # # # # # # #


    def discriminatorDropout(self):
        global disDropout
        if self.disDropoutCheckbox.get() == 1:
            disDropout = True
        elif self.disDropoutCheckbox.get() == 0:
            disDropout = False


                        # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Model Parameter Check    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # #


    def disVarCheck(self):
        global disNormEps, disNormAffine
        self.textConsoleLog.delete("0.0", "end")
        self.textConsoleLog.insert("0.0", "GAN's Discriminator Architecture")
        self.textConsoleLog.insert("end", "\nNumber of layers: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{discriminator_layers}")
        self.textConsoleLog.insert("end", "\nCore Type: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{discriminator_core_layer}")

        if discriminator_core_layer == "Convolution Layer":
            global disKernelSize, disStrideSize, disPaddingSize
            disKernelSize = self.disConvKernelSize.get()
            disStrideSize = self.disConvStrideSize.get()
            disPaddingSize = self.disConvPaddingSize.get()

            self.textConsoleLog.insert("end", "\nKernel Size: ", "textconfig_green")
            self.textConsoleLog.insert("end", f"{disKernelSize}")
            self.textConsoleLog.insert("end", "\nStride Size: ", "textconfig_green")
            self.textConsoleLog.insert("end", f"{disStrideSize}")
            self.textConsoleLog.insert("end", "\nPadding Size: ", "textconfig_green")
            self.textConsoleLog.insert("end", f"{disPaddingSize}")
        
        elif discriminator_core_layer == "Dense Layer":
            self.textConsoleLog.insert("end", "\nDense Layer Additive Bias: ", "textconfig_green")
            self.textConsoleLog.insert("end", f"{disBias}")

        self.textConsoleLog.insert("end", "\n\nAdditional LSTM Layer: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{disLSTM}")
        self.textConsoleLog.insert("end", "\nLSTM Bias: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{disLSTMBias}")
        self.textConsoleLog.insert("end", "\nBiDirectional LSTM: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{disLSTMBi}")
        self.textConsoleLog.insert("end", "\nLSTM Dropout probability: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{disLSTMDrop}")

        disNormEps = self.disNormEpsSize.get()

        self.textConsoleLog.insert("end", "\n\nNormalization Layer Type: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{discriminator_norm_layer}")
        self.textConsoleLog.insert("end", "\nNumerical Stability Epsilon: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{disNormEps}")        
        self.textConsoleLog.insert("end", "\nLearnable Affine Parameters: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{disNormAffine}")

        self.textConsoleLog.insert("end", "\n\nActivation Layer Type: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{discriminator_activation_layer}")

        if discriminator_activation_layer == "ReLU":
            self.textConsoleLog.insert("end", "\nIn-place Tensors: ", "textconfig_green")
            self.textConsoleLog.insert("end", f"{disActivationInplace}")        
        elif discriminator_activation_layer == "LeakyReLU":
            self.textConsoleLog.insert("end", "\nNegative Slope: ", "textconfig_green")
            self.textConsoleLog.insert("end", f"{disActivationNegativeSlope}")    
        elif discriminator_activation_layer == "ELU":
            self.textConsoleLog.insert("end", "\nAlpha Slope: ", "textconfig_green")
            self.textConsoleLog.insert("end", f"{disActivationAlpha}")    

        self.textConsoleLog.insert("end", "\n\nDropout Layer: ", "textconfig_green")
        self.textConsoleLog.insert("end", f"{disDropout}")

    def disLayerCounter(self, num: str):
        global discriminator_layers
        discriminator_layers = int(num)
        self.disLayerLabel.configure(text=f"Number of hidden layers: {discriminator_layers}")


              # # # # # # # # # # # # # # # #
    # # # # # #    Training Loop Buttons    # # # # # # # # # # # # # # # #
              # # # # # # # # # # # # # # # #
   

    def generatedFrames(self):
        global save_generated_frames
        if self.saveGeneratedFramesCheckbox.get() == 1:
            save_generated_frames = True
        elif self.saveGeneratedFramesCheckbox.get() == 0:
            save_generated_frames = False

    def plotModelLoss(self):
        global plot_loss
        if self.plotLossCheckbox.get() == 1:
            plot_loss = True
        elif self.plotLossCheckbox.get() == 0:
            plot_loss = False

    def plotModelPredictions(self):
        global plot_preds
        if self.plotPredsCheckbox.get() == 1:
            plot_preds = True
        elif self.plotPredsCheckbox.get() == 0:
            plot_preds = False

    def plotGenerated(self):
        global plot_frames
        if self.plotFramesCheckbox.get() == 1:
            plot_frames = True
        elif self.plotFramesCheckbox.get() == 0:
            plot_frames = False

    def evalMSECheck(self):
        global eval_mse
        if self.evalMSECheckbox.get() == 1:
            eval_mse = True
        elif self.evalMSECheckbox.get() == 0:
            eval_mse = False

    def evalPSNRCheck(self):
        global eval_psnr
        if self.evalPSNRCheckbox.get() == 1:
            eval_psnr = True
        elif self.evalPSNRCheckbox.get() == 0:
            eval_psnr = False

    def evalL1Check(self):
        global eval_l1
        if self.evalL1Checkbox.get() == 1:
            eval_l1 = True
        elif self.evalL1Checkbox.get() == 0:
            eval_l1 = False

    def deterministicAlgorithms(self):
        global deterministic_algorithms
        if self.trainingDeterministicAlgorithms.get() == 1:
            deterministic_algorithms = True
        elif self.trainingDeterministicAlgorithms.get() == 0:
            deterministic_algorithms = False

    def trainingLossFunction(self, new: str):
        global loss_function
        loss_function = new

    def trainingOptimizer(self, new: str):
        global optimizer
        optimizer = new

        self.learningRateLabel = tk.CTkLabel(self.tabview.tab("Training"), text="Learning Rate:")
        self.learningRateLabel.grid(row=15, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.learningRate = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="float e.g. 0.0002")
        self.learningRate.grid(row=16, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.beta1Label = tk.CTkLabel(self.tabview.tab("Training"), text="Beta 1:")
        self.beta1Label.grid(row=17, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.beta1 = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="float e.g. 0.5")
        self.beta1.grid(row=18, column=0, padx=20, pady=(5, 10), sticky="nsew")

        self.beta2Label = tk.CTkLabel(self.tabview.tab("Training"), text="Beta 2:")
        self.beta2Label.grid(row=19, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.beta2 = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="float e.g. 0.999")
        self.beta2.grid(row=20, column=0, padx=20, pady=(5, 10), sticky="nsew")

    def modelSaveType(self, new: str):
        global model_save_type
        model_save_type = new

        self.saveCountLabel = tk.CTkLabel(self.tabview.tab("Training"), text="How many:")
        self.saveCountLabel.grid(row=26, column=0, padx=20, pady=(5, 10), sticky="nsew")
        self.saveCount = tk.CTkEntry(self.tabview.tab("Training"), placeholder_text="int e.g. 50")
        self.saveCount.grid(row=27, column=0, padx=20, pady=(5, 10), sticky="nsew")

    def trainingVarCheck(self):
        global seed, batch_size, latent_dim, epochs, threads, lr, beta1, beta2, model_save_count, batches_per_frame_save
        self.textConsoleLog.delete("0.0", "end")
        self.textConsoleLog.insert("0.0", "GAN's Training Loop Architecture")

        seed = self.trainingSeed.get()
        self.textConsoleLog.insert("end", "\nRandom Seed: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{seed}")
        self.textConsoleLog.insert("end", "\nDeterministic Algorithms: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{deterministic_algorithms}")

        batch_size = self.batchSize.get()
        self.textConsoleLog.insert("end", "\nBatch Size: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{batch_size}")

        latent_dim = self.latentDim.get()
        self.textConsoleLog.insert("end", "\nLatent Dimension: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{latent_dim}")

        epochs = self.numEpochs.get()
        self.textConsoleLog.insert("end", "\nNumber of Epochs: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{epochs}")

        threads = self.workerThreads.get()
        self.textConsoleLog.insert("end", "\nNumber of Threads: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{threads}")
        self.textConsoleLog.insert("end", "\nLoss Function Type: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{loss_function}")
        self.textConsoleLog.insert("end", "\nOptimizer Type: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{optimizer}")

        lr = self.learningRate.get()
        self.textConsoleLog.insert("end", "\nLearning Rate: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{lr}")

        beta1 = self.beta1.get()
        self.textConsoleLog.insert("end", "\nBeta1: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{beta1}")

        beta2 = self.beta2.get()
        self.textConsoleLog.insert("end", "\nBeta2: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{beta2}")
        self.textConsoleLog.insert("end", "\nSave Generated Frames: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{save_generated_frames}")

        batches_per_frame_save = self.batchesPerSave.get()
        self.textConsoleLog.insert("end", "\nAfter how many batches: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{batches_per_frame_save}")
        self.textConsoleLog.insert("end", "\nSave models in: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{model_save_type}")

        model_save_count = self.saveCount.get()
        self.textConsoleLog.insert("end", "\nHow many: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{model_save_count}")
        self.textConsoleLog.insert("end", "\nPlot Model Loss: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{plot_loss}")
        self.textConsoleLog.insert("end", "\nPlot Model Predictions: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{plot_preds}")
        self.textConsoleLog.insert("end", "\nPlot Generated Frames: ", "textconfig_blue")
        self.textConsoleLog.insert("end", f"{plot_frames}")


                        # # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Creating Frames Logic    # # # # # # # # # # #
                        # # # # # # # # # # # # # # # #
   

    def example_frames(self):
        global interpolated_frames, original_frames, interpolated_frames_idx, original_frames_idx
        iframes = glob.glob('src/vimeo_septuplet/sequences/00016/0016/*.png')
        if iframes:
            interpolated_frames = []
            interpolated_frames_idx, original_frames_idx = 0, 0
        for frame in iframes:
            interpolated_frames.append(Image.open(frame))
        original_frames = [interpolated_frames[0], interpolated_frames[2], interpolated_frames[4], interpolated_frames[6]]
        self.leftimage = tk.CTkImage(light_image=original_frames[0], size=(448, 256))
        self.leftimagelabel = tk.CTkLabel(self.labelImageLoad, image=self.leftimage, text="")
        self.leftimagelabel.grid(row=1, column=0)
        self.rightimage = tk.CTkImage(light_image=interpolated_frames[0], size=(448, 256))
        self.rightimagelabel = tk.CTkLabel(self.labelImageLoad, image=self.rightimage, text="")
        self.rightimagelabel.grid(row=1, column=1)

        self.leftimagetext.configure(text=f"Original frames ({original_frames_idx+1}/{len(original_frames)})")
        self.rightimagetext.configure(text=f"Interpolated frames ({interpolated_frames_idx+1}/{len(interpolated_frames)})")

    def load_generator(self):
        global generator_class_file_path, generator_model_file_path
        generator_model_file_path = filedialog.askopenfilename(title="Select a generator model", filetypes=[("PTH files", "*.pth")])
        generator_class_file_path = filedialog.askopenfilename(title="Select a python script", filetypes=[("Python files", "*.py")])
        self.textConsoleLog.insert("end", f"Selected generator {generator_model_file_path} which uses the class {generator_class_file_path}.", )

    def generate_frames(self):
        global interpolated_frames, original_frames, interpolated_frames_idx, original_frames_idx
        if generator_class_file_path != "":
            self.textConsoleLog.delete("0.0", "end")
            module_name = 'generator_module'
            spec = importlib.util.spec_from_file_location(module_name, generator_class_file_path)
            module = importlib.util.module_from_spec(spec)
            print(module)
            spec.loader.exec_module(module)
            Generator = getattr(module, 'Generator', None)
            if Generator:
                try:
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                except Exception as e:
                    device = torch.device("cpu")
                generator = Generator().to(device)
                generator = nn.DataParallel(generator)
                generator.load_state_dict(torch.load(generator_model_file_path))
                generator.eval()
                original_frames, interpolated_frames = Generate(generator, device)
                interpolated_frames_idx, original_frames_idx = 0, 0
                self.leftimage = tk.CTkImage(light_image=original_frames[0], size=(448, 256))
                self.leftimagelabel = tk.CTkLabel(self.labelImageLoad, image=self.leftimage, text="")
                self.leftimagelabel.grid(row=1, column=0)
                self.rightimage = tk.CTkImage(light_image=interpolated_frames[0], size=(448, 256))
                self.rightimagelabel = tk.CTkLabel(self.labelImageLoad, image=self.rightimage, text="")
                self.rightimagelabel.grid(row=1, column=1)

                self.leftimagetext.configure(text=f"Original frames ({original_frames_idx+1}/{len(original_frames)})")
                self.rightimagetext.configure(text=f"Interpolated frames ({interpolated_frames_idx+1}/{len(interpolated_frames)})")

            else:
                self.textConsoleLog.insert("end", "Failed to initialise the generator. Make sure the filename is generator.py and the class is called Generator")
    
    def frame_GIF(self):
        print(type(interpolated_frames[0]))
        try:
            global generated_fps
            generated_fps = int(self.gifFPS.get())
        except:
            pass
        try:
            out_dir = os.path.join(os.path.expanduser('~'), 'Downloads', 'generated.gif')
            interpolated_frames[0].save(
                out_dir,
                format='GIF',
                append_images=interpolated_frames[1:],
                duration=int(1000/generated_fps),
                save_all=True,
                loop=0
            )

        except:
            self.textConsoleLog.insert("end", "No frames detected, try generating frames first.")


                        # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Display Frame Logic    # # # # # # # # # # #
                        # # # # # # # # # # # # # # #
    
    
    def backwardFrames(self):
        global interpolated_frames_idx, original_frames_idx
        if interpolated_frames_idx == 0 and original_frames_idx == 0:
            fake_len = len(interpolated_frames)
            real_len = len(original_frames)
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=original_frames[real_len-1], size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=interpolated_frames[fake_len-1], size=(448, 256)))
            interpolated_frames_idx = fake_len - 1
            original_frames_idx = real_len - 1
        elif interpolated_frames_idx != 0 and original_frames_idx == 0:
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=interpolated_frames[interpolated_frames_idx-1], size=(448, 256)))
            interpolated_frames_idx = interpolated_frames_idx - 1
        elif interpolated_frames_idx != 0 and original_frames_idx != 0:
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=original_frames[original_frames_idx-1], size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=interpolated_frames[interpolated_frames_idx-1], size=(448, 256)))
            interpolated_frames_idx = interpolated_frames_idx - 1
            original_frames_idx = original_frames_idx - 1

        self.leftimagetext.configure(text=f"Original frames ({original_frames_idx+1}/{len(original_frames)})")
        self.rightimagetext.configure(text=f"Interpolated frames ({interpolated_frames_idx+1}/{len(interpolated_frames)})")
        
    def forwardFrames(self):
        global interpolated_frames_idx, original_frames_idx
        if interpolated_frames_idx == len(interpolated_frames) - 1 and original_frames_idx == len(original_frames) - 1:
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=original_frames[0], size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=interpolated_frames[0], size=(448, 256)))
            interpolated_frames_idx = 0
            original_frames_idx = 0
        elif interpolated_frames_idx != len(interpolated_frames) - 1 and original_frames_idx == len(original_frames) - 1:
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=interpolated_frames[interpolated_frames_idx+1], size=(448, 256)))
            interpolated_frames_idx = interpolated_frames_idx + 1
        elif interpolated_frames_idx != len(interpolated_frames) - 1 and original_frames_idx != len(original_frames) - 1:
            self.leftimagelabel.configure(image=tk.CTkImage(light_image=original_frames[original_frames_idx+1], size=(448, 256)))
            self.rightimagelabel.configure(image=tk.CTkImage(light_image=interpolated_frames[interpolated_frames_idx+1], size=(448, 256)))
            interpolated_frames_idx = interpolated_frames_idx + 1
            original_frames_idx = original_frames_idx + 1

        self.leftimagetext.configure(text=f"Original frames ({original_frames_idx+1}/{len(original_frames)})")
        self.rightimagetext.configure(text=f"Interpolated frames ({interpolated_frames_idx+1}/{len(interpolated_frames)})")
        
        
                        # # # # # # # # # # # # # # #
    # # # # # # # # # # #    Training Loop Logic    # # # # # # # # # # #
                        # # # # # # # # # # # # # # #
                        

    def beginTraining(self):
        self.textConsoleLog.delete("0.0", "end")
        self.textConsoleLog.insert("0.0", "Saving model parameters...")
        
        self.saveParams()

        self.textConsoleLog.insert("1.0", "Parameters saved", "textconfig_yellow")
        self.textConsoleLog.insert("2.0", "Loading dataset and models...")
        
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(deterministic_algorithms)

        train_data, training_batches, training_remainder, device, generator, discriminator, criteron, optimizer_generator, optimizer_discriminator = train_model.start()
        
        self.textConsoleLog.insert("3.0", "Parameters saved", "textconfig_yellow")

        generated_frames = []
        g_losses = []
        g_mean_preds = []
        d_losses = []
        d_fake_mean_preds = []
        d_real_mean_preds = []
        psnr_calc = PeakSignalNoiseRatio()
        mse = []
        psnr = []
        l1 = []
        iteration = 0
        epoch_count = 0
        real_label = 1.0
        fake_label = 0.0

        for epoch in range(epochs):
            epoch_count += 1
            for batch_idx, data in enumerate(train_data):
                iteration += 1
                startt = time.time()
                try:
                    self.textConsoleLog.delete("4.0", "end")
                except:
                    pass

                self.textConsoleLog.insert("4.0", f"Epoch {epoch_count}/{epochs} Batch {batch_idx+1}/{training_batches} ({((batch_idx+1+(training_batches*(epoch_count-1)))/(training_batches*epochs))*100:.2f}%)")
                #print(f"Epoch {epoch_count}/{epochs} Batch {batch_idx+1}/{training_batches} ({((batch_idx+1+(training_batches*(epoch_count-1)))/(training_batches*epochs))*100:.2f}%)", end='\r', flush=True)
                frames, label = data     
                frame_list = []
                
                try:
                    for i in range(batch_size):
                        for j in range(len(frames)):
                            frame_list.append(frames[j][i])
                except IndexError:
                    print(f"IndexError, using remainder batch size at batch {batch_idx}")
                        
                frame_initial = []
                frame_mid = []
                frame_after = []
                for i in range(len(frame_list) - 1):
                    if i+2 < len(frame_list) and 0 < ((i+1) % 7) < 6:
                        frame_initial.append(frame_list[i])
                        frame_mid.append(frame_list[i+1])
                        frame_after.append(frame_list[i+2])
                    else:
                        pass

                frame_initial = torch.stack(frame_initial).to(device).contiguous()
                frame_mid = torch.stack(frame_mid).to(device)
                frame_after = torch.stack(frame_after).to(device).contiguous()
                frame_list_tensor = torch.stack(frame_list).to(device)
                frame_pairs = torch.stack((frame_initial, frame_after)).to(device).contiguous()

                # # # # # # # # # # # # # # # #
                #    Training Discriminator   #
                # # # # # # # # # # # # # # # #

                discriminator.zero_grad()
                optimizer_discriminator.zero_grad()
                
                # All real data
                #print("Disc with real")
                if (batch_idx+1) % training_batches == 0:
                    label = torch.full((training_remainder*len(frames),), real_label, dtype=torch.float, device=device)
                else:
                    label = torch.full((batch_size*len(frames),), real_label, dtype=torch.float, device=device)
                #print(len(label))
                real_output = discriminator(frame_list_tensor).view(-1)
                #print(len(real_output))
                d_real_error = criteron(real_output, label)
                d_real_error.backward()
                d_real_mean_pred = real_output.mean().item()

                # All fake data
                #print("Disc with fake")
                if (batch_idx+1) % training_batches == 0:
                    label = torch.full((training_remainder*(len(frames)-2),), fake_label, dtype=torch.float, device=device)
                else:
                    label = torch.full((batch_size*(len(frames)-2),), fake_label, dtype=torch.float, device=device)
                #noise = torch.randn(batch_size*len(frames), latent_dim, 1, 1, device=device)
                #fake_frames = generator(noise)
                fake_frames = generator(frame_pairs[0], frame_pairs[1])
                fake_output = discriminator(fake_frames.detach()).view(-1)
        
                d_fake_error = criteron(fake_output, label)
                d_fake_error.backward()
                d_fake_mean_pred = fake_output.mean().item()
                
                d_error = d_real_error + d_fake_error
                #print("Optimizing disc parameters")
                optimizer_discriminator.step()
                #print("Gen")
                
                # # # # # # # # # # # # # #
                #    Training Generator   #
                # # # # # # # # # # # # # #

                generator.zero_grad()
                optimizer_generator.zero_grad()
                #print("Gen with fake")
                label.fill_(real_label) # Generator aims to have its data classified as real
                output = discriminator(fake_frames).view(-1)
                g_error = criteron(output, label)
                g_error.backward()
                g_mean_pred = output.mean().item()
                #print("Optimizing gen parameters")
                optimizer_generator.step()

                g_losses.append(g_error.item())
                d_losses.append(d_error.item())
                g_mean_preds.append(g_mean_pred)
                d_fake_mean_preds.append(d_fake_mean_pred)
                d_real_mean_preds.append(d_real_mean_pred)
                #print("Appended results")

                if save_generated_frames == True:
                    if (iteration % batches_per_frame_save == 0) or ((epoch == epochs) and (i == len(train_data)-1)):
                        with torch.no_grad():
                            fake = generator(frame_pairs[0], frame_pairs[1]).detach().cpu()
                        generated_frames.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=4))
                    torch.cuda.empty_cache()

                if model_save_type == "Batches":
                    if (iteration % model_save_count == 0) or (iteration % training_batches == 0):
                        torch.save(generator.state_dict(), f'models/vimeo/generator_{epoch_count}_{batch_idx}.pth')
                        torch.save(discriminator.state_dict(), f'models/vimeo/discriminator_{epoch_count}_{batch_idx}.pth')
                elif model_save_type == "Epochs":
                    if (epoch_count % model_save_count == 0) or (iteration % training_batches == 0):
                        torch.save(generator.state_dict(), f'models/vimeo/generator_{epoch_count}_{batch_idx}.pth')
                        torch.save(discriminator.state_dict(), f'models/vimeo/discriminator_{epoch_count}_{batch_idx}.pth')

                if (iteration % 50 == 0) or (iteration % training_batches == 0):
                    mse.append(F.mse_loss(fake.cpu(), frame_mid.cpu()))
                    psnr.append(psnr_calc(fake.cpu(), frame_mid.cpu()))
                    l1.append(F.l1_loss(fake.cpu(), frame_mid.cpu()))
        
        endt = time.time()
        self.textConsoleLog.insert("end", "\nFinished training after: ")
        self.textConsoleLog.insert("end", format_time(startt, endt, device))
        torch.save(generator.state_dict(), f'models/vimeo/generator_final.pth')
        torch.save(discriminator.state_dict(), f'models/vimeo/discriminator_final.pth')

        if plot_loss:
            plt.figure(figsize=(10,5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(g_losses,label="G")
            plt.plot(d_losses,label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig('plots/loss.png', dpi=300, bbox_inches='tight')
            plt.show()

        if plot_preds:
            plt.figure(figsize=(10,5))
            plt.title("All Mean Predictions")
            plt.plot(g_mean_preds,label="G")
            plt.plot(d_fake_mean_preds,label="D-Fake")
            plt.plot(d_real_mean_preds,label="D-Real")
            plt.xlabel("iterations")
            plt.ylabel("Probability")
            plt.legend()
            plt.savefig('plots/preds.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(10,5))
            plt.title("G Mean Predictions")
            plt.plot(g_mean_preds,label="G")
            plt.xlabel("iterations")
            plt.ylabel("Probability")
            plt.legend()
            plt.savefig('plots/g_preds.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(10,5))
            plt.title("D Real Mean Predictions")
            plt.plot(d_real_mean_preds,label="D-Real")
            plt.xlabel("iterations")
            plt.ylabel("Probability")
            plt.legend()
            plt.savefig('plots/d_real_preds.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(10,5))
            plt.title("D Fake Mean Predictions")
            plt.plot(d_fake_mean_preds,label="D-Fake")
            plt.xlabel("iterations")
            plt.ylabel("Probability")
            plt.legend()
            plt.savefig('plots/d_fake_preds.png', dpi=300, bbox_inches='tight')
            plt.show()

        if plot_loss:
            plt.figure(figsize=(10,5))
            plt.title("MSE-Loss")
            plt.plot(mse)
            plt.xlabel("iterations")
            plt.ylabel("MSE-Loss")
            plt.legend()
            plt.savefig('plots/mse.png', dpi=300, bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(10,5))
            plt.title("PSNR-Loss")
            plt.plot(psnr)
            plt.xlabel("iterations")
            plt.ylabel("PSNR-Loss")
            plt.legend()
            plt.savefig('plots/psnr.png', dpi=300, bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(10,5))
            plt.title("L1-Loss")
            plt.plot(l1)
            plt.xlabel("iterations")
            plt.ylabel("L1-Loss")
            plt.legend()
            plt.savefig('plots/l1.png', dpi=300, bbox_inches='tight')
            plt.show()

        if save_generated_frames:
            for i in range(len(generated_frames)):
                plt.figure(figsize=(8,8))
                plt.title(f"Set {i+1} of Generated Frames")
                plt.axis("off")
                plt.imshow(generated_frames[i].permute(1, 2, 0).numpy())
                plt.savefig(f'plots/generated{i+1}.png', dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Plotted {i+1}/{len(generated_frames)} ({((i+1)/(len(generated_frames))*100):.2f}%)", end='\r', flush=True)

    def saveParams(self):
        config = configparser.ConfigParser()
        config['generator'] = {
            'generator_layers': str(generator_layers),
            'generator_core_layer': str(generator_core_layer),
            'generator_norm_layer': str(generator_norm_layer),
            'generator_activation_layer': str(generator_activation_layer),
            'genLSTM': str(genLSTM),
            'genLSTMBias': str(genLSTMBias),
            'genLSTMBi': str(genLSTMBi),
            'genLSTMDrop': str(genLSTMDrop),
            'genBias': str(genBias),
            'genNormEps': str(genNormEps),
            'genNormAffine': str(genNormAffine),
            'genDropout': str(genDropout),
            'genActivationInplace': str(genActivationInplace),
            'genActivationNegativeSlope': str(genActivationNegativeSlope),
            'genActivationAlpha': str(genActivationAlpha),
            'genKernelSize': str(genKernelSize),
            'genStrideSize': str(genStrideSize),
            'genPaddingSize': str(genPaddingSize)
            }
        
        config['discriminator'] = {
            'discriminator_layers': str(discriminator_layers),
            'discriminator_core_layer': str(discriminator_core_layer),
            'discriminator_norm_layer': str(discriminator_norm_layer),
            'discriminator_activation_layer': str(discriminator_activation_layer),
            'disLSTM': str(disLSTM),
            'disLSTMBias': str(disLSTMBias),
            'disLSTMBi': str(disLSTMBi),
            'disLSTMDrop': str(disLSTMDrop),
            'disBias': str(disBias),
            'disNormEps': str(disNormEps),
            'disNormAffine': str(disNormAffine),
            'disDropout': str(disDropout),
            'disActivationInplace': str(disActivationInplace),
            'disActivationNegativeSlope': str(disActivationNegativeSlope),
            'disActivationAlpha': str(disActivationAlpha),
            'disKernelSize': str(disKernelSize),
            'disStrideSize': str(disStrideSize),
            'disPaddingSize': str(disPaddingSize)
            }
        
        config['training'] = {
            'seed': str(seed),
            'deterministic_algorithms': str(deterministic_algorithms),
            'batch_size': str(batch_size),
            'latent_dim': str(latent_dim),
            'epochs': str(epochs),
            'threads': str(threads),
            'loss_function': str(loss_function),
            'optimizer': str(optimizer),
            'lr': str(lr),
            'beta1': str(beta1),
            'beta2': str(beta2),
            'save_generated_frames': str(save_generated_frames),
            'batches_per_frame_save': str(batches_per_frame_save),
            'model_save_type': str(model_save_type),
            'model_save_count': str(model_save_count),
            'plot_loss': str(plot_loss),
            'plot_preds': str(plot_preds),
            'plot_frames': str(plot_frames),
        }
        
        with open('params.ini', 'w') as configfile:
            config.write(configfile)

    def appearanceChange(self, new: str):
        tk.set_appearance_mode(new) #Updates GUI theme

    def scalingChange(self, new: str):
        newFloat = int(new.replace("%", "")) / 100
        tk.set_widget_scaling(newFloat) #Updates GUI scaling

if __name__ == "__main__":
    app = App() #Creates new app instance
    app.mainloop() #Keeps app running by using a loop
