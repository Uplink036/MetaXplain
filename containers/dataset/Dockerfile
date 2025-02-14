FROM python
RUN mkdir /tmp/data
WORKDIR /loader
COPY . .
ENV DATABASE_URL=mongodb://localhost:27017/
ENV SCRUB_DB=0
CMD ["./run.sh"]