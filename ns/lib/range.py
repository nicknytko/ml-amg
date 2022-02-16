class Range:
    def __init__(self, low, high):
        assert(low <= high)
        self.low = low
        self.high = high

    def overlaps(self, other):
        if other.low >= self.low and other.low <= self.high:
            return True
        if other.high >= self.low and other.high <= self.high:
            return True
        return False

    def contains(self, other):
        if other.low >= self.low and other.high <= self.high:
            return True
        return False

    def extend(self, other):
        '''
        Extends the current range if combining the two ranges results in a single, unbroken range
        '''
        if self.contains(other):
            return True
        elif self.overlaps(other):
            self.low = min(self.low, other.low)
            self.high = max(self.high, other.high)
            return True
        else:
            return False

    def __repr__(self):
        return f'<Range [{self.low}, {self.high}]>'


class DisjointRanges:
    def __init__(self, ranges=[]):
        self.ranges = ranges.copy()
        self.ranges.sort(key=lambda r: r.low)
        self._combine_ranges()

    def _combine_ranges(self):
        # Always assume ranges are sorted in ascending order by "low" value
        i = 0
        while i < len(self.ranges) - 1:
            if self.ranges[i].extend(self.ranges[i+1]):
                del self.ranges[i+1]
                # We combined with next range, stay at this index
            else:
                i += 1

    def _bisect(self, x):
        '''
        Binary search, taken from bisect library
        '''
        low = 0
        high = len(self.ranges)
        while low < high:
            mid = (low + high)//2
            if x.low < self.ranges[mid].low:
                high = mid
            else:
                low = mid+1
        return low

    def _insort(self, x):
        idx = self._bisect(x)
        self.ranges.insert(idx, x)

    def extend(self, rng):
        self._insort(rng)
        self._combine_ranges()

    def __repr__(self):
        return f'<DisjointRanges {self.ranges}>'
