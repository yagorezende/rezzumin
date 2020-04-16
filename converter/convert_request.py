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
        with open("{}/{}_abs.txt".format(self.path, self.id), 'w') as destination:
            destination.write(abs)