import os
import numpy as np
import scipy as spi
import json

CLEAN_FOLDER_PATH = 'measurements_dataset//clean'
COLUMNS_NUMBER = 5
DATA_PART = 1

class DataSegmentator:
    def __init__(self, segment_length):
        self.segment_length = segment_length    # length of one segment in ms    
        self.metadata = []                      # list of metadata dictionaries
        self.timestamps = []                    # list of timestamps arrays
        self.adc_data = []                      # list of adc_data arrays

    def iterate_files(self):
        files = os.listdir(CLEAN_FOLDER_PATH)
        for file_name in files:
            self.process_file(file_name)

    def process_file(self, filename):
        file = open(CLEAN_FOLDER_PATH + '//' + filename, "r")
        lines = []
        line = file.readline()
        while line:
            lines.append(line)
            line = file.readline()
        file.close()

        size = len(lines)
        metadata = dict()
        timestamps = np.zeros([size-1], dtype=int)
        adc_data = np.zeros([size-1, COLUMNS_NUMBER], dtype=float)

        for index in range(size):
            line = lines[index]
            if index == 0:
                metadata = json.loads(line)
            else:
                data = json.loads(line)
                timestamps[index-1] = data.get('timestamp')
                for i in range(COLUMNS_NUMBER):
                    adc_data[index-1][i] = data.get('adc_outputs')[i]
        
        self.metadata.append(metadata)
        self.timestamps.append(timestamps)
        self.adc_data.append(adc_data)
    
    def get_metadata_clean(self, metadata):
        result = np.zeros_like(metadata)
        previous_ids = []

        for i in range(len(metadata)):
            ids = metadata[i].get('labels').get('person_data').get('person_id')
            if previous_ids.count(ids) == 0:
                result[i] = len(previous_ids)
                previous_ids.append(ids)
            else:
                result[i] = previous_ids.index(ids)
        return result
    
    def resample_segments(self, segments, prop=1):
        indices = [i for i in range(len(segments))]
        resampled_segments = []

        indices = np.random.choice(indices, size=int(len(indices)/prop), replace=False)
        for i in range(len(indices)):
            resampled_segments.append(segments[indices[i]])
        return resampled_segments

    def fixed_apply_segmentation(self):
        if self.metadata == [] or self.timestamps == [] or self.adc_data == []:
            raise Exception('segmentation tried on empty arrays :c')
        meta = self.get_metadata_clean(self.metadata)
        length_counter = dict()
        all_segments = []
        for file_index in range(len(self.metadata)):
            total_segments = int(np.ceil(self.timestamps[file_index][-1]/self.segment_length))
            #print(total_segments)
            segment_index = 0
            #for segment_index in range(total_segments):
            i = 0
            while True:
                segment = []
                seg_row = [ 0 for i in range(6)]
                #segment = np.zeros([self.segment_length+2, COLUMNS_NUMBER+1], dtype=float)
                seg_row[0] = float(meta[file_index])
                #segment[0][0] = float(meta[file_index])

                segment.append(seg_row)
                seg_start = segment_index*self.segment_length
                seg_end = seg_start + self.segment_length

                if seg_end >= self.timestamps[file_index][-1]:
                    seg_end = self.timestamps[file_index][-1] - 1
                if seg_start > seg_end:
                    break

                #print(seg_start, seg_end, total_segments, self.timestamps[file_index][-2], self.timestamps[file_index][-1])

                #ind = 1
                #i = 0
                #for i in range(total_segments):
                while True:
                    if (self.timestamps[file_index][i]) > seg_end:         # taking into consideration not fully aligned measurements
                        if length_counter.get(len(segment)) is not None:
                            length_counter[len(segment)] += 1
                        else:
                            length_counter[len(segment)] = 1
                        break

                    seg_row = [ 0 for k in range(6)]
                    if (self.timestamps[file_index][i]) >= seg_start:       # taking into consideration not fully aligned measurements
                        seg_row[0] = float(self.timestamps[file_index][i])
                        #segment[ind][0] = float(self.timestamps[file_index][i])
                        for j in range(COLUMNS_NUMBER):
                            seg_row[j+1] = self.adc_data[file_index][i][j]
                            #segment[ind][j+1] = self.adc_data[file_index][i][j]
                    segment.append(seg_row)
                    #ind += 1
                    #print(self.timestamps[file_index][i], seg_end)


                    i += 1
                #print(len(segment))
                #todo - normalize segments
                #print(i)
                
                if len(segment) >= int(self.segment_length/100+1):
                    all_segments.append(segment.copy())
                segment_index += 1
        print(length_counter)
        #all_segments = self.normalize_segments(all_segments)
        all_segments = self.spline_normalize_segments(all_segments, int(np.ceil(self.segment_length / 100 + 1)))
        all_segments = self.resample_segments(all_segments, DATA_PART) # prop is for deciding how big of a part of dataset will be evaluated
        return all_segments, self.metadata

    def spline_normalize_segments(self, segments, length):
        new_segments = []
        for s in segments:
            value = np.zeros((length+1, 6))
            seg_meta = np.array(s)
            value[0] = seg_meta[0]
            seg = np.delete(seg_meta, 0, 0)

            old = np.linspace(0,1,len(seg))
            new = np.linspace(0,1, length)
            for i in range(0, 5):
                column = np.take(seg, i, 1)
                f = spi.interpolate.interp1d(old, column, kind='cubic')(new)
                for j in range(0, length):
                    value[j+1][i] = f[j]
            new_segments.append(value.copy())
        return new_segments


    def normalize_segments(self, segments):
        #first line of each segment contains metadata
        #segments should be 14 elements long
        new_segments = []
        for s in segments:
            #pass
            value = np.zeros((int(self.segment_length/100 + 1), 6))
            seg = np.array(s)
            #print(np.size(seg, 0), s[-1][0]-s[1][0])
            #if s[-1][0]-s[1][0] < self.segment_length and s[-1][0]-s[1][0] > self.segment_length-100:
            #if s[-1][0]-s[1][0] > self.segment_length-100:
            if True:
                value[0] = seg[0]
                i = 1
                while True:
                    dist = seg[i][0] - seg[1][0]    # distance between timestamps
                    if dist%100 == 0:               # perfectly normalised distance
                        value[i] = seg[i]
                        value[i][0] = (i-1)*100
                    elif dist < (i-1)*100:
                        if i+1 < len(seg):
                            value[i] = seg[i] + ((i-1)*100 - seg[i][0])/(seg[i+1][0] - seg[i][0])*(seg[i+1] - seg[i])
                        else:
                            value[i] = seg[i-2] + (seg[i-1] - seg[i-2])*((i-1)*100 - seg[i-2][0])/(seg[i-1][0] - seg[i-2][0])
                        value[i][0] = (i-1)*100
                    else:
                        value[i] = seg[i] - (seg[i][0] - (i-1)*100)/(seg[i][0] - seg[i-1][0])*(seg[i] - seg[i-1])
                        value[i][0] = (i-1)*100
                    if i*100 == self.segment_length: # todo - powinno byc o jedno dluzsze
                        break
                    i += 1
                new_segments.append(value.copy())
        return new_segments

    def apply_segmentation(self):
        if self.metadata == [] or self.timestamps == [] or self.adc_data == []:
            raise Exception('segmentation tried on empty arrays :c')
        meta = self.get_metadata_clean(self.metadata)

        count = len(self.metadata)
        all_segments = []
        for index in range(count):
            # index will be used as a label for segments
            total_segments = int(np.ceil(self.timestamps[index][-1]/self.segment_length))
            # first column contains timestamp and the following ones contain adc_data
            # first row contains index in timestamp column and 0 in remaining columns
            for segment_index in range(total_segments):
                segment = np.zeros([self.segment_length+2, COLUMNS_NUMBER+1], dtype=float)
                segment[0][0] = float(meta[index])
                seg_start = segment_index*self.segment_length
                seg_end = seg_start + self.segment_length
                if seg_end >= len(self.timestamps[index]):
                    seg_end = len(self.timestamps[index]) - 1
                if seg_start > seg_end:
                    break
                ind = 1
                for i in range(seg_start, seg_end+1):
                    segment[ind][0] = float(self.timestamps[index][i])
                    for j in range(COLUMNS_NUMBER):
                        segment[ind][j+1] = self.adc_data[index][i][j]
                    ind += 1
                if segment[-1][0] != 0:
                    all_segments.append(segment.copy())
        
        all_segments = self.resample_segments(all_segments)
        return all_segments, self.metadata


        
