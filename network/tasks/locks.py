from django.core.cache import cache

LOCK_EXPIRE = 60 * 10


class Lock:
    def __init__(self, lock_key, lock_value=True):

        if not bool(lock_value):
            raise ValueError('Lock value cannot evaluate to False.')

        self.lock_key = lock_key
        self.lock_value = lock_value
        self.added = False

    def add(self):
        return cache.add(self.lock_key, self.lock_value, LOCK_EXPIRE)

    def delete(self):
        return cache.delete(self.lock_key)

    def exists(self):
        return bool(cache.get(self.lock_key))


class ExpRecUpdateQueueLock(Lock):
    def __init__(self, experiment):
        lock_key = self.create_lock_key(experiment)
        super().__init__(lock_key)

    def create_lock_key(self, experiment):
        return '-'.join([
            'experiment-recommendation-update-queue',
            str(experiment.pk),
        ])


class ExpRecUpdateExecuteLock(Lock):
    def __init__(self, experiment):
        lock_key = self.create_lock_key(experiment)
        super().__init__(lock_key)

    def create_lock_key(self, experiment):
        return '-'.join([
            'experiment-recommendation-update-execute',
            str(experiment.pk),
        ])


class UserRecUpdateQueueLock(Lock):
    def __init__(self, following, followed):
        lock_key = self.create_lock_key(following, followed)
        super().__init__(lock_key)

    def create_lock_key(self, following, followed):
        return '-'.join([
            'user-recommendation-update-queue',
            str(following.pk),
            str(followed.pk),
        ])


class UserRecUpdateExecuteLock(Lock):
    def __init__(self, following, followed):
        lock_key = self.create_lock_key(following, followed)
        super().__init__(lock_key)

    def create_lock_key(self, following, followed):
        return '-'.join([
            'user-recommendation-update-execute',
            str(following.pk),
            str(followed.pk),
        ])
