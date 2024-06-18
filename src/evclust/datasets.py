# -*- coding: utf-8 -*-
# This file as well as the whole evclust package are licenced under the MIT licence (see the LICENCE.txt)
# Armel SOUBEIGA (armelsoubeiga.github.io), France, 2023

"""
This module contains all tests datasets
"""

# ---------------------- Packges------------------------------------------------
import pathlib
import pandas as pd

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"


# ---------------------- Data 1-------------------------------------------------
def load_decathlon():
    """The Decathlon dataset from FactoMineR."""

    decathlon = pd.read_csv(DATASETS_DIR / "decathlon.csv")
    decathlon.columns = ["athlete", *map(str.lower, decathlon.columns[1:])]
    decathlon.athlete = decathlon.athlete.apply(str.title)
    decathlon = decathlon.set_index(["competition", "athlete"])
    return decathlon


# ---------------------- Data 2-------------------------------------------------
def load_iris():
    """Iris data."""
    return pd.read_csv(DATASETS_DIR / "iris.csv")


# ---------------------- Data 3-------------------------------------------------
def load_letters():
    return pd.read_csv(DATASETS_DIR / "lettersIJLDavidson.csv")


# ---------------------- Data 4-------------------------------------------------
def load_seeds():
    return pd.read_csv(DATASETS_DIR / "seeds.csv")


def load_forest():
    return pd.read_csv(DATASETS_DIR / "forest_type.csv")


def load_thyroid():
    return pd.read_csv(DATASETS_DIR / "thyroid.csv")


def load_libras():
    return pd.read_csv(DATASETS_DIR / "libras.csv", header=None)


def load_2d_dataset():
    return pd.read_csv(DATASETS_DIR / "2c2dDataset.csv", header=None)


def load_2c6d_dataset():
    return pd.read_csv(DATASETS_DIR / "2c6dDataset.csv", header=None)


def load_crescent2D():
    return pd.read_csv(DATASETS_DIR / "crescent2D.csv", header=None)


def load_toys3c2d():
    return pd.read_csv(DATASETS_DIR / "toys3c2d.csv", header=None)

def load_toys3c2d_2():
    return pd.read_csv(DATASETS_DIR / "toys3c2d_2.csv", header=None)
