FROM python
WORKDIR loader
COPY . .
ENV DATABASE_URL=mongodb://localhost:27017/
CMD ["./run.sh"]