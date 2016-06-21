from django.core.management.base import BaseCommand, CommandError
from results import models
from unipath import Path
import pickle
from pydub import AudioSegment
from PIL import Image, ImageDraw
import re

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
            results = pickle.load(f)
            print results.keys()
            print results['stressful']
            classification = []
            for i in range(len(results['windows'])):
                classification.append((results['windows'][i][0],
                    results['windows'][i][1], results['stressful'][i]))
        print classification

        # Make the name a bit nicer
        p = re.compile(r'([^_]+)_(\d+)-(\d+)-(\d+)T(\d+):(\d+):(\d+)')
        m = p.match(wav.stem)
        if m:
            filename = '{} {}-{}-{} {}:{}:{}'.format(m.group(1), m.group(2),
                m.group(3), m.group(4), m.group(5), m.group(6), m.group(7))
            print filename
        else:
            filename = wav.stem

        length = classification[-1][END] # end of the last window
        stress = sum([w[STRESS] for w in classification]) / float(len(classification)) * 100
        rec = models.Recording(filename=filename, length=length, stressful=stress,
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

        #stress_thres = max(0.5000001, (max_stress - min_stress) / 2.0 + min_stress)
        stress_thres =      (max_stress - min_stress)*0.50 + min_stress
        high_stress_thres = (max_stress - min_stress)*0.75 + min_stress

        avg_stress = sum(stress_hist) / float(len(stress_hist))
        print "Minimum:", min_stress
        print "Maximum:", max_stress
        print "Average:", avg_stress
        print "Threshold:", stress_thres

        sounds = {}
        tot_sounds = 0
        curr_sound = 0
        curr_loud_sound = 0
        in_sound = False
        in_loud_sound = False
        same_start = False
        for lvl in range(len(stress_hist)):
            low_started = False
            low_ended = False
            if stress_hist[lvl] > stress_thres :
                if in_sound == False:
                    low_started = True
                    tot_sounds += 1
                    curr_sound = tot_sounds
                    sounds[curr_sound] = {'windows': [], 'stress': []}
                in_sound = True
                sounds[curr_sound]['windows'].append(lvl)
                sounds[curr_sound]['stress'].append(stress_hist[lvl])
            else:
                low_ended = True
                in_sound = False


            if stress_hist[lvl] > high_stress_thres :
                if low_ended == True :
                    
                if in_loud_sound == False:
                    if low_started :
                        same_start = True
                    else : 
                        same_start = False

                    tot_sounds += 1
                    curr_loud_sound = tot_sounds
                    sounds[curr_loud_sound] = {'windows': [], 'stress': []}
                elif low_started :
                    same_start = False

                in_sound = True
                sounds[curr_loud_sound]['windows'].append(lvl)
                sounds[curr_loud_sound]['stress'].append(stress_hist[lvl])
            else:
                in_sound = False

        print sounds

        for key, s in sounds.items():
            start = classification[min(s['windows'])][BEGIN]
            end   = classification[max(s['windows'])][END]
            stress = sum(s['stress']) / float(len(s['stress'])) * 100
            sound = models.Sound(recording=rec, start=start, end=end,
                stressful=stress, relaxing=0)
            sound.save()

        print "Creating image file"
        self._wav_to_image(wav, filename)

    def _wav_to_image(self, filename, outname):
        width = 1000
        height = 100
        barwidth = 1
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        audio = AudioSegment.from_file(filename)
        audio.split_to_mono()
        duration = audio.duration_seconds
        max_amp = audio.max_dBFS

        dBpp = -90 / height # dB per pixel

        sec_width = (duration * 1000) / width * barwidth
        for i in range(width / barwidth):
            sec = audio[int(i*sec_width):int((i+1)*sec_width)]
            amp = sec.max_dBFS * dBpp
            draw.rectangle([(i, height-amp), ((i+1)*barwidth, amp)], fill="#2196F3")
        del draw

        dest_file = Path('static/' + outname + '.png')
        img.save(dest_file)
