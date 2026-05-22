import redis

r = redis.Redis(host='localhost', port=6379)

r.set("test", "working")

print(r.get("test"))