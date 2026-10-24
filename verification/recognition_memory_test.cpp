#include "recognitionmemory.h"
#include <iostream>
#include <stdexcept>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
facecore::RecognitionResult known(const char* id) {
    facecore::RecognitionResult r; r.state=facecore::RecognitionState::Known;r.personId=id;return r;
}
int main() {
    try {
        TrackRecognitionMemory m;
        require(m.due(1000000,1000),"new track must be recognized");
        m.update(known("001"),1,1000000);
        require(!m.cached(),"one vote must not confirm a person");
        require(!m.due(1000000,1000),"same frame timestamp must not cause repeated inference");
        require(!m.due(1249999,1000)&&m.due(1250000,1000),"pending track must retry at 250ms");
        m.update(known("001"),2,1250000);m.update(known("001"),3,1500000);
        require(m.cached()&&m.cached()->personId=="001","three agreeing votes must confirm");
        require(!m.due(2499999,1000)&&m.due(2500000,1000),"known identity must be rechecked at 1s");
        facecore::RecognitionResult unknown;unknown.state=facecore::RecognitionState::Unknown;
        m.update(unknown,4,2500000);
        require(!m.cached(),"contradictory unknown must immediately hide old identity");
        m.update(unknown,5,2750000);m.update(unknown,6,3000000);
        require(m.cached()&&m.cached()->state==facecore::RecognitionState::Unknown,"unknown consensus");
        require(!m.due(3249999,1000)&&m.due(3250000,1000),"unknown must continue retrying");
        m.update(known("002"),7,3250000);
        require(!m.cached(),"old unknown votes must not lock out a known person");
        m.update(known("002"),8,3500000);m.update(known("002"),9,3750000);
        require(m.cached()&&m.cached()->personId=="002","unknown-to-known recovery");
        m.update(known("001"),10,4750000);
        require(!m.cached(),"track reuse must hide previous person's label");
        m.update(known("001"),11,5000000);m.update(known("001"),12,5250000);
        require(m.cached()&&m.cached()->personId=="001","track reuse must accept new votes");
        facecore::RecognitionResult error;error.state=facecore::RecognitionState::Error;
        m.update(error,13,6250000);m.update(error,14,6500000);m.update(error,15,6750000);
        require(!m.cached()&&m.votes.empty(),"errors must not create unknown votes or retain stale labels");
        require(!m.due(6999999,1000)&&m.due(7000000,1000),"inference errors must be rate limited");
        require(m.due(6000000,1000),"backward clock adjustment must not freeze inference");
        m.update(known(""),16,7000000);
        require(m.votes.empty(),"empty known identity must not become an unknown vote");
        TrackRecognitionMemory mixed;
        mixed.update(known("001"),1,1000000);mixed.update(known("002"),2,1250000);
        mixed.update(known("001"),3,1500000);mixed.update(known("002"),4,1750000);
        require(!mixed.cached(),"two votes each must not confirm");
        mixed.update(known("001"),5,2000000);
        require(mixed.cached()&&mixed.cached()->personId=="001","3/5 majority must still work");
        std::cout<<"PASS: voting, periodic verification, unknown recovery, track reuse, inference failures, clock changes\n";
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
