from django.core.management.base import BaseCommand, CommandError
from results import models
from unipath import Path
import pickle
from pydub import AudioSegment
from PIL import Image, ImageDraw

BEGIN  = 0
END    = 1
STRESS = 2
RELAX  = 3

class Command(BaseCommand):
    help = "Add a sound file to the database"

    def add_arguments(self, parser):
        parser.add_argument('wav')
        parser.add_argument('classification')

    def handle(self, *args, **options):
        wav = Path(options['wav'])
        class_file = Path(options['classification'])
        with open(class_file, 'r') as f:
            classification = pickle.load(f)

        length = classification[-1][END] # end of the last window
        rec = models.Recording(filename=wav.stem, length=length, stressful=0,
            relaxing=0)
        rec.save()
        print "Loaded file", rec, "with length", length

        max_stress = 0.0
        min_stress = 1.0
        avg_stress = 0.0
        stress_hist = []
        for w in classification:
            if w[STRESS] > max_stress:
                max_stress = w[STRESS]
            if w[STRESS] < min_stress:
                min_stress = w[STRESS]
            stress_hist.append(w[STRESS])

        stress_thres = min(0.2, (max_stress - min_stress) / 2.0)

        avg_stress = sum(stress_hist) / float(len(classification))
        print "Minimum:", min_stress
        print "Maximum:", max_stress
        print "Average:", avg_stress

        self._wav_to_image(wav)

    def _wav_to_image(self, filename):
        width = 1000
        height = 100
        barwidth = 1
        img = Image.new('RGB', (width, height), "#2196F3")
        draw = ImageDraw.Draw(img)

        audio = AudioSegment.from_file(filename)
        audio.split_to_mono()
        duration = audio.duration_seconds
        max_amp = audio.max_dBFS

        dBpp = -90 / height # dB per pixel

        sec_width = (duration * 1000) / width * barwidth
        for i in range(width / barwidth):
            sec = audio[int(i*sec_width):int((i+1)*sec_width)]
            amp = sec.dBFS * dBpp
            draw.rectangle([(i, 0), ((i+1)*barwidth, amp)], fill=(255, 255, 255))
        del draw

        dest_file = Path('static/' + filename.stem + '.png')
        img.save(dest_file)
