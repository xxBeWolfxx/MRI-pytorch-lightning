import time

class Bar():
    length = 100
    decimal = 1
    progress = 0
    def __init__(self, total) -> None:
        self.totalLimit = total

    def process(self, newValue):
        self.progress = self.progress + newValue
        return self.printing()

    def printing(self):
        if self.progress >= self.totalLimit:
            percent = 100.0
            donePart = self.length
            remainPart = 0

        else:
            percent = self.progress * 100 / self.totalLimit
            percent = round(percent, 1)
            donePart = int(self.length * percent / 100)
            remainPart = self.length - donePart
        
        bar = donePart * '#' + '-' * remainPart

        barTerminal = f'\rProgress: |{bar}| {percent}%'
        print(barTerminal, end = '\r')

        if self.progress >= self.totalLimit:
            print(f'\rProgress: |{bar}| {percent}%   |PROCESS DONE|\n')
            return 1

        return 0


barek = Bar(120)

for x in range(100):
    if barek.process(5):
        break
    time.sleep(1)