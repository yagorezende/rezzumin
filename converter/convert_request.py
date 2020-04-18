class ConvertRequest:
    def __init__(self, id, root_path):
        self.id = id
        self.pdfFile = None
        self.path = root_path
        self.fullText = None
        self.isSaved = False
        self.isPdf = False
        self.abs = None
        self.body = None
        self.status = 0
        self.portion = 0

    def getPDFName(self):
        return "{}.pdf".format(self.id)

    def getTextName(self):
        return "{}.txt".format(self.id)

    def getPDFLocation(self):
        return "{}/{}".format(self.path, self.getPDFName())

    def getTextLocation(self):
        return "{}/{}".format(self.path, self.getTextName())

    def cleanupText(self):
        ab_init = self.fullText.find("Abstract")
        ab_end = self.fullText[ab_init:].find(".\n\n") + ab_init
        # print("ab_init:{} ab_end:{}".format(ab_init, ab_end))
        # print("abs = "+self.fullText[ab_init:ab_end])

        res_init = self.fullText.find("Resumo")
        res_end = self.fullText[res_init:].find(".\n\n") + res_init
        # print("res_init:{} res_end:{}".format(res_init, res_end))
        # print("res = " + self.fullText[res_init:res_end])

        if res_end > ab_end:
            self.body = self.fullText[res_end:]
        else:
            self.body = self.fullText[ab_end:]
        self.abs = self.fullText[res_init:res_end]

        # remove line breaks
        plain_text = ''
        for line in self.body:
            plain_text += line.strip('\n')
        self.body = plain_text

    def savePDFFile(self) -> bool:
        if self.pdfFile is None:
            return False
        with open(self.getPDFLocation(), 'wb+') as destination:
            for chunk in self.pdfFile.chunks():
                destination.write(chunk)
        return True

    def saveAbs(self, abs):
        with open("{}/{}_abs.txt".format(self.path, self.id), 'w', encoding='utf-8') as destination:
            destination.write(abs)

    def getAbsPath(self) -> str:
        return "{}/{}_abs.txt".format(self.path, self.id)

    def loadStatus(self) -> int:
        with open("{}/{}_status.txt".format(self.path, self.id), 'r', encoding='utf-8') as destination:
            self.status = eval(destination.read())
            return self.status

    def saveStatus(self):
        with open("{}/{}_status.txt".format(self.path, self.id), 'w', encoding='utf-8') as destination:
            destination.write(str(self.status))

    def incrementStatus(self, hop):
        self.status += hop
        self.saveStatus()