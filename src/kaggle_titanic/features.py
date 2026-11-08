"""Reusable feature-engineering helpers."""

import pandas as pd

TITLE_RE = r",\s*([^.]+)\."  # "Surname, Title. Given names" -> "Title"

# Decisions logged in JOURNAL.md (2026-11-08).
TITLE_GROUPS = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    # French / modern forms -> English equivalents
    "Mlle": "Miss",
    "Mme": "Mrs",
    "Ms": "Miss",
    # Nobility, professional and military -> Fancy
    "Sir": "Fancy",
    "Lady": "Fancy",
    "the Countess": "Fancy",
    "Jonkheer": "Fancy",
    "Don": "Fancy",
    "Dona": "Fancy",
    "Dr": "Fancy",
    "Rev": "Fancy",
    "Col": "Fancy",
    "Major": "Fancy",
    "Capt": "Fancy",
}


def extract_title(names: pd.Series) -> pd.Series:
    """Raw title from Name, e.g. 'Braund, Mr. Owen Harris' -> 'Mr'."""
    return names.str.extract(TITLE_RE, expand=False).str.strip()


class AgeImputer:
    """Fill missing Age with the median for the passenger's Pclass x TitleGroup.

    Empty cells back off to the TitleGroup median, then the overall median.
    Fit on training rows only, then transform any frame (avoids leakage).
    Expects a 'TitleGroup' column (see title_group) and 'Pclass'.
    Chosen by scripts/age_imputation_test.py; decision in JOURNAL.md.
    """

    def fit(self, df: pd.DataFrame) -> "AgeImputer":
        self.overall_ = df["Age"].median()
        self.by_title_ = df.groupby("TitleGroup")["Age"].median()
        self.by_class_title_ = df.groupby(["Pclass", "TitleGroup"])["Age"].median()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Imputed age for every row (ignores the row's own Age)."""
        keys = ["Pclass", "TitleGroup"]
        pred = df.join(self.by_class_title_.rename("_pred"), on=keys)["_pred"]
        pred = pred.fillna(df["TitleGroup"].map(self.by_title_))
        return pred.fillna(self.overall_)

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Age with missing values filled in."""
        return df["Age"].fillna(self.predict(df))


def title_group(names: pd.Series) -> pd.Series:
    """Collapsed title: Mr / Mrs / Miss / Master / Fancy.

    Any title not in TITLE_GROUPS (e.g. one that only appears in future data)
    falls back to 'Fancy' rather than becoming NaN.
    """
    return extract_title(names).map(TITLE_GROUPS).fillna("Fancy")


def family_size(df: pd.DataFrame) -> pd.Series:
    """SibSp + Parch + 1 (the passenger themself). Decision in JOURNAL.md."""
    return df["SibSp"] + df["Parch"] + 1


def fare_per_family(df: pd.DataFrame) -> pd.Series:
    """Fare / FamilySize: an approximate per-person price.

    Fare is the price of the whole *ticket*, repeated for everyone on it.
    Dividing by FamilySize assumes ticket group == declared family, which
    is not always true (e.g. friends or servants on a family's ticket).
    """
    return df["Fare"] / family_size(df)


def ticket_group_size(df: pd.DataFrame, all_passengers: pd.DataFrame) -> pd.Series:
    """Number of passengers sharing this row's Ticket.

    Counted over `all_passengers` (train + test), because members of one group
    can be split across the two files. Uses features only (no Survived), so it
    is not target leakage.
    """
    counts = all_passengers["Ticket"].value_counts()
    return df["Ticket"].map(counts).fillna(1).astype(int)


def fare_per_person(df: pd.DataFrame, all_passengers: pd.DataFrame) -> pd.Series:
    """Fare / ticket group size: the price per passenger on the ticket."""
    return df["Fare"] / ticket_group_size(df, all_passengers)


class FarePerPersonImputer:
    """Fill missing FarePerPerson with the median for the passenger's Pclass x Embarked.

    Backs off to the Pclass median, then the overall median.
    Fit on training rows only, then transform any frame.
    Expects 'FarePerPerson', 'Pclass' and 'Embarked' columns.
    Decision in JOURNAL.md: median 3rd-class Southampton fare for the one test
    passenger (1044, travelling alone) with no Fare.
    """

    def fit(self, df: pd.DataFrame) -> "FarePerPersonImputer":
        self.overall_ = df["FarePerPerson"].median()
        self.by_class_ = df.groupby("Pclass")["FarePerPerson"].median()
        self.by_class_port_ = df.groupby(["Pclass", "Embarked"])["FarePerPerson"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        keys = ["Pclass", "Embarked"]
        pred = df.join(self.by_class_port_.rename("_pred"), on=keys)["_pred"]
        pred = pred.fillna(df["Pclass"].map(self.by_class_)).fillna(self.overall_)
        return df["FarePerPerson"].fillna(pred)


def cabin_known(df: pd.DataFrame) -> pd.Series:
    """1 if a Cabin is recorded, else 0. Decision in JOURNAL.md (H6)."""
    return df["Cabin"].notna().astype(int)


def deck(df: pd.DataFrame) -> pd.Series:
    """Deck letter = first character of Cabin; 'U' (unknown) where Cabin is missing.

    'T' (a single passenger in train, none in test) is merged into 'U'.
    Decision in JOURNAL.md.
    """
    return df["Cabin"].str[0].fillna("U").replace({"T": "U"})


class EmbarkedImputer:
    """Fill missing Embarked with the port whose median FarePerPerson, within the
    passenger's Pclass, is closest to the passenger's own FarePerPerson.

    Fit on training rows only. Expects 'Embarked', 'Pclass', 'FarePerPerson'.
    Decision in JOURNAL.md.
    """

    def fit(self, df: pd.DataFrame) -> "EmbarkedImputer":
        self.medians_ = df.groupby(["Pclass", "Embarked"])["FarePerPerson"].median()
        return self

    def _closest_port(self, row: pd.Series) -> str:
        ports = self.medians_.loc[row["Pclass"]]
        return (ports - row["FarePerPerson"]).abs().idxmin()

    def transform(self, df: pd.DataFrame) -> pd.Series:
        out = df["Embarked"].copy()
        missing = out.isna()
        out[missing] = df[missing].apply(self._closest_port, axis=1)
        return out
