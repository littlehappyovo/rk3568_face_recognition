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
        require(m.due(1000000,50),"new track must be recognized");
        m.update(known("001"),1,1000000);
        require(!m.cached(),"one vote must not confirm a person");
        require(!m.due(1000000,1000),"same frame timestamp must not cause repeated inference");
        require(!m.due(1049999,50)&&m.due(1050000,50),"pending track must retry at 50ms");
        m.update(known("001"),2,1050000);m.update(known("001"),3,1100000);
        require(m.cached()&&m.cached()->personId=="001","three agreeing votes must confirm");
        require(!m.due(1000000000,50),"confirmed identity must not be rechecked while its track remains active");
        facecore::RecognitionResult unknown;unknown.state=facecore::RecognitionState::Unknown;
        m.update(unknown,4,2100000);
        require(!m.cached(),"contradictory unknown must immediately hide old identity");
        m.update(unknown,5,2150000);m.update(unknown,6,2200000);
        require(!m.cached(),"unknown votes must not label a track unknown during the grace period");
        require(!m.due(2249999,50)&&m.due(2250000,50),"unresolved track must continue retrying at 50ms");
        m.update(known("002"),7,2250000);
        require(!m.cached(),"old unknown votes must not lock out a known person");
        m.update(known("002"),8,2300000);m.update(known("002"),9,2350000);
        require(m.cached()&&m.cached()->personId=="002","known identity must still reach consensus during grace");
        m.update(known("001"),10,3350000);
        require(!m.cached(),"track reuse must hide previous person's label");
        m.update(known("001"),11,3400000);m.update(known("001"),12,3450000);
        require(m.cached()&&m.cached()->personId=="001","track reuse must accept new votes");
        facecore::RecognitionResult error;error.state=facecore::RecognitionState::Error;
        m.update(error,13,4450000);m.update(error,14,4500000);m.update(error,15,4550000);
        require(!m.cached()&&m.votes.empty(),"errors must not create unknown votes or retain stale labels");
        require(!m.due(4599999,50)&&m.due(4600000,50),"inference errors must be rate limited at 50ms");
        require(m.due(4000000,50),"backward clock adjustment must not freeze inference");
        m.update(known(""),16,4600000);
        require(m.votes.empty(),"empty known identity must not become an unknown vote");
        TrackRecognitionMemory mixed;
        mixed.update(known("001"),1,1000000);mixed.update(known("002"),2,1050000);
        mixed.update(known("001"),3,1100000);mixed.update(known("002"),4,1150000);
        require(!mixed.cached(),"two votes each must not confirm");
        mixed.update(known("001"),5,1200000);
        require(mixed.cached()&&mixed.cached()->personId=="001","3/5 majority must still work");

        TrackRecognitionMemory unknownGrace;
        unknownGrace.update(unknown,1,1000000);
        unknownGrace.update(unknown,2,1050000);
        unknownGrace.update(unknown,3,1100000);
        require(!unknownGrace.cached(),"three unknown votes must remain unresolved before three seconds");
        require(unknownGrace.due(1150000,50),"unresolved unknown must continue periodic inference");
        unknownGrace.update(unknown,4,3999999);
        require(!unknownGrace.cached(),"unknown must remain unresolved just before the three-second boundary");
        unknownGrace.update(unknown,5,4000000);
        require(unknownGrace.cached()&&unknownGrace.cached()->state==facecore::RecognitionState::Unknown,
                "unknown consensus may be exposed once three seconds have elapsed");
        require(unknownGrace.due(4050000,50),"confirmed unknown must continue retrying at the configured interval");
        std::cout<<"PASS: voting, 3-second unknown grace, periodic verification, unknown recovery, track reuse, inference failures, clock changes\n";
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
