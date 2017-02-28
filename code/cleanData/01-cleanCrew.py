# -*- coding: utf-8 -*-

import nltk
import nltk.data
import nltk.tag
import os
import re
import json
import string
from compiler.ast import flatten
from glob import glob
import datetime


def prepForLDA(filename, outPath):
    """Main function to clean raw scraped data """
    #os.chdir(inPath)
    jsonData = loadJSON(filename, details=True)
    data = dictPull(jsonData, 'data')
    data = removeSctnHdr(filename, data)
    data = removeURL(data)
    # data=removeHTML(data)
    cntries = dictPull(jsonData, 'name')

    time()
    stClean = removePunct(data)
    stClean = tokenize(stClean)
    stClean = remNouns(stClean)
    stClean = remACR(stClean)
    stClean = remCommonWords(stClean, cntries)
    stClean = remNum(stClean)
    stClean = lemmatize(stClean)
    stClean = remCommonWords(stClean, cntries)
    time()

    jsonDataFin = updateDict(jsonData, stClean)
    jsonDataFin = cleanDict(jsonDataFin)
    os.chdir(outPath)
    saveJSON(jsonDataFin, filename, outPath, details=True, newName=True)

# Helper functions


def time():
    print '\t\t' + datetime.datetime.now().time().isoformat()


def loadJSON(file, details=False):
    """Load json file"""
    if details:
        print 'Cleaning Data for ' + file + '...\n'
    return json.load(open(file, 'rb'))


def dictPull(dataDict, key):
    """Pull out specific values
    from a dictionary into a list"""
    info = []
    for dat in dataDict:
        info.append(dat[key].encode('utf-8'))
    return info


def removeSctnHdr(filename, data):
    """Remove source specific section headers"""
    if filename.split('_')[0] == 'FH':
        return [re.sub('\n.*?:&nbsp;', ' ', dat)
                for dat in data]
    elif filename.split('_')[0] == 'StateHR':
        data = [re.sub('\nPDF.*?SUMMARYShare', ' ', dat, flags=re.DOTALL)
                for dat in data]
        data = [re.sub('\n\n\t\t.*?\n\n', ' ', dat, flags=re.DOTALL)
                for dat in data]
        for ii in range(0, len(data)):
            for letter in string.ascii_lowercase:
                data[ii] = re.sub('\n' + letter + '..*?\n', ' ',
                                  data[ii], flags=re.DOTALL)
        return data
    elif filename.split('_')[0] == 'StateRF':
        data = [re.sub('\nPDF.*?SummaryShare', ' ', dat, flags=re.DOTALL)
                for dat in data]
        data = [re.sub('\n\n\nSection.*?&nbsp;\n', ' ', dat, flags=re.DOTALL)
                for dat in data]
        return data
    else:
        return data


def noURL(string):
    """Identify URLs in text"""
    return re.sub(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        '',
        string)


def removeURL(data):
    """Remove URLs from text"""
    return [noURL(dat) for dat in data]


def removeHTML(data):
    """Clean up html"""
    return [nltk.clean_html(dat) for dat in data]


def removePunct(stories):
    """Remove punctuation and html leftovers"""
    repPunct = string.maketrans(
        string.punctuation, ' ' * len(string.punctuation))
    storiesNoPunct = [story.translate(repPunct) for story in stories]
    storiesNoPunct = [re.sub(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', story)
        for story in storiesNoPunct]
    otherPunct = ['nbsp', 'lsquo', 'rsquo', 'ldquo', 'rdquo', 'quot', 'eacute']
    for slash in otherPunct:
        storiesNoPunct = [story.replace(slash, " ")
                          for story in storiesNoPunct]
    print('\tPunctuation removed...')
    return storiesNoPunct


def tokenize(stories):
    """Create tokens from text"""
    storiesToken = [[word for word in story.split()] for story in stories]
    print('\tTokenized...')
    return storiesToken


def remWrd(stories, wrds, keepWrds=True):
    """Fast way to remove items in list from
    another list"""
    nStories = []
    for ii in range(0, len(stories)):
        story = stories[ii]
        if len(wrds) == len(stories):
            sWrds = set(wrds[ii]).intersection(story)
        else:
            sWrds = set(wrds).intersection(story)
        if keepWrds:
            nStory = [word for word in story if word in sWrds]
        else:
            nStory = [word for word in story if word not in sWrds]
        nStories.append(nStory)
    return nStories


def remNouns(stories):
    """remmove upper case words"""
    print('\tUpper case words removed...')
    return [[word for word in story if not word[0].isupper()]
            for story in stories]


def propUpper(string):
    """Helper to remove acronyms"""
    return sum(1. for l in string if l.isupper()) / len(string)


def remACR(stories):
    """Remove acronyms and make all words lowercase"""
    storiesNoACR = [[word for word in story if propUpper(word) < 0.5]
                    for story in stories]
    storiesLower = [[word.lower() for word in story]
                    for story in storiesNoACR]
    print('\tAcronyms removed & tokens now lower cased...')
    return storiesLower


def remCommonWords(stories, cntryNames):
    """Remove common, irrelevant words"""
    remove = nltk.corpus.stopwords.words('english')
    cntCurr = ['verdean', 'ariary', 'kyat', 'mexican', 'chinese', 'yen',
               'uzbekistani', 'cayman', 'dollar', 'lankan', 'belarusian',
               'kong', 'macedonian', 'jordanian', 'cambodian', 'lilangeni',
               'polish', 'taiwan', 'mozambican', 'icelandic', 'marka', 'lats',
               'turkish', 'balboa', 'ugandan', 'iraqi', 'hryvnia', 'won',
               'moldovan', 'dalasi', 'somali', 'zloty', 'lira', 'east',
               'argentine', 'guatemalan', 'real', 'seychellois', 'indonesian',
               'hungarian', 'egyptian', 'koruna', 'kwanza', 'vanuatu',
               'comorian', 'sol', 'netherlands', 'vietnamese', 'moroccan',
               'malawian', 'new', 'bermudian', 'cape', 'rufiyaa', 'special',
               'states', 'maldivian', 'rupiah', 'mauritanian', 'algerian',
               'bosnia', 'helena', 'ouguiya', 'kenyan', 'omani',
               'kr\xc3\xb3na', 'manat', 'angolan', 'tom\xc3\xa9', 'surinamese',
               'saudi', 'serbian', 'metical', 'croatian', 'uruguayan', 'dong',
               'vatu', 'franc', 'czech', 'malaysian', 'nepalese', 'japanese',
               'cfa', 'nigerian', 'leonean', 'honduran', 'nicaraguan', 'cfp',
               'c\xc3\xb3rdoba', 'riel', 'israeli', 'caribbean', 'estonian',
               'guyanese', 'birr', 'tobago', 'cedi', 'loti', 'indian',
               'ukrainian', 'afghani', 'lari', 'aruban', 'tugrik', 'colombian',
               'malagasy', 'brunei', 'lithuanian', 'forint', 'baht', 'lek',
               'nuevo', 'bolivian', 'liberian', 'armenian', 'lev',
               'paraguayan', 'bhutanese', 'tanzanian', 'belize', 'naira',
               'renminbi', 'kazakhstani', 'chilean', 'central', 'kroon',
               'canadian', 'albanian', 'guarani', 'united', 'british', 'papua',
               'barbadian', 'kuna', 'bahraini', 'antillean', 'latvian',
               'romanian', 'islands', 'namibian', 'south', 'leu', 'libyan',
               'sri', 'panamanian', 'ethiopian', 'denar', 'colon', 'escudo',
               'congolese', 'myanma', 'ringgit', 'rican', 'swazi', 'trinidad',
               'gulden', 'west', 'uae', 'nakfa', 'shilling', 'bangladeshi',
               'bolivar', 'gibraltar', 'saint', 'brazilian', 'australian',
               'dirham', 'rand', 'sheqel', 'krona', 'bahamian', 'bulgarian',
               'tunisian', 'krone', 'ruble', 'georgian', 'sierra', 'eritrean',
               'turkmen', 'somoni', 'russian', 'gambian', 'kyrgyzstani',
               'haitian', 'litas', 'ghanaian', 'lebanese', 'pakistani',
               'dominican', 'iranian', 'boliviano', 'pula', 'riyal', 'peso',
               'solomon', 'jamaican', 'pataca', 'yemeni', 'and', 'pound',
               'danish', 'peruvian', 'herzegovina', 'philippine', 'zambian',
               'lesotho', 'tenge', 'sudanese', 'mongolian', 'rights',
               'burundi', 'north', 'syrian', 'botswana', 'swedish', 'florin',
               'kuwaiti', 'guinean', 'som', 'macanese', 'djiboutian', 'samoan',
               's\xc3\xa3o', 'taka', 'norwegian', 'kina', 'konvertibilna',
               'rwandan', 'azerbaijani', 'zimbabwean', 'korean', 'falkland',
               'rial', 'european', 'tajikistani', 'lao', 'singapore',
               'mauritian', 'kip', 'hong', 'kwacha', 'fijian', 'dram',
               'gourde', 'qatari', 'afghan', 'euro', 'zealand', 'costa',
               'cuban', 'pr\xc3\xadncipe', 'quetzal', 'ngultrum', 'venezuelan',
               'rupee', 'thai', 'drawing', 'dinar', 'lempira', 'tala',
               'african', 'swiss', 'dobra', 'leone']
    remove.extend(
        ([x.lower() for x in cntryNames],
         cntCurr,
         'document', 'end', 'web', 'facto', 'examining', 'compared',
         'whereabouts', 'inspectorate', 'examination', 'year', 'years',
         'month', 'months', 'day', 'days', 'january', 'february', 'march',
         'april', 'may', 'june', 'july', 'august', 'september', 'october',
         'november', 'december', 'one', 'two', 'three', 'four', 'five', 'six',
         'seven', 'eight', 'nine', 'ten', 'grand', 'north', 'east', 'south',
         'west', 'southeast', 'eastern', 'southern', 'northern', 'department',
         'see', 'findings', 'new', 'old', 'men', 'man', 'sun', 'eye', 'ear',
         'cut', 'although', 'though', 'country', 'received', 'report',
         'edition', 'ombudsman', 'gendarme', 'gendarmerie', 'carabineros',
         'vez', 'dina', 'maquiladora', 'nassara', 'gacaca', 'territory',
         'province', 'provincial', 'federal', 'mainland', 'canton', 'cantonal',
         'island', 'county', 'principality', 'prefecture', 'governorates',
         'governorate', 'oblast', 'oblasts', 'municipality', 'directorate',
         'district', 'region', 'riyal', 'franc', 'euro', 'shilling', 'dirham',
         'rial', 'russian', 'lira', 'ruble', 'dinar', 'peso', 'rupee',
         'koruna', 'dollar', 'peseta', 'reais', 'ylang', 'emalangeni',
         'ethiopian', 'macedonian', 'colombian', 'japanese', 'kunas',
         'restaveks', 'restavek', 'bidoon', 'cordoba', 'kanun', 'sjambok',
         'cocaleros', 'bateyes', 'takrima', 'lobola', 'african', 'philippine',
         'emirate', 'djamena', 'zquez', 'dji', 'kou', 'ndez', 'ada', 'soum',
         'nchez', 'sub', 'lamibe', 'iar', 'los', 'url', 'aire', 'antilles',
         'ath', 'aire', 'birr', 'del', 'indigenous', 'neo', 'lei', 'dust',
         'non', 'bin', 'kou', 'rez'))
    remove = flatten(remove)
    storiesNoStop = remWrd(stories, remove, keepWrds=False)
    storiesNoStop = [[word for word in story if len(word) > 2]
                     for story in storiesNoStop]
    print('\tStop words removed...')
    return storiesNoStop


def remNum(stories):
    """Remove tokens that are numbers"""
    storiesNoNum = [[word for word in story if not word.isdigit()]
                    for story in stories]
    print('\tNumbers removed...')
    return storiesNoNum


def lemmatize(stories):
    """Lemmatize"""
    wnl = nltk.stem.WordNetLemmatizer()
    storiesLemm = [[wnl.lemmatize(word) for word in story]
                   for story in stories]
    print('\tLemmatized...')
    return storiesLemm


def updateDict(jsonListDict, storiesClean):
    """Update dictionary with new data as key, value"""
    for ii in range(0, len(jsonListDict)):
        jsonListDict[ii]['dataClean'] = storiesClean[ii]
    return jsonListDict


def cleanDict(jsonListDict):
    """Remove empty items in dictionary"""
    return [x for x in jsonListDict if len(x['dataClean']) != 0]


def saveJSON(data, filename, out_path, details=False, newName=False):
    """save data as json"""
    if details:
        print '\n Data for ' + filename + ' cleaned \n'
    if newName:
        split = os.path.split(filename)
        filename = os.path.join(out_path, split[1])
        #filename = filename.split('.')[0] + '_Clean.json'
    f = open(filename, 'wb')
    json.dump(data, f, sort_keys=True)
    f.close()


def _get_data(dir_path, path):
    """Private function to get the absolute path to the installed files."""
    cwd = os.path.abspath(os.path.dirname(__file__))
    joined = os.path.join(dir_path, path)
    out_dir = os.path.join(cwd, joined)
    return out_dir


def main():
    data_dir = _get_data('../../data/raw', '*json')
    out_dir = _get_data('../../data', 'cleaned')
    files = glob(data_dir)

    # Clean each file
    for f in files:
        prepForLDA(f, out_dir)


if __name__ == '__main__':
    main()
